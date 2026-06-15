#![doc = include_str!("../README.md")]
// The pyo3 0.22 `#[pyfunction]` macro (under edition 2024) generates an unsafe
// trampoline that omits inner `unsafe` blocks and a redundant `PyErr` conversion,
// tripping `unsafe_op_in_unsafe_fn` and `clippy::useless_conversion`. These fire
// on macro-generated code only; our hand-written unsafe uses explicit
// `unsafe { .. }` blocks, so allowing them crate-wide hides nothing real.
#![allow(unsafe_op_in_unsafe_fn, clippy::useless_conversion)]

//! Polars integration for tokmat, usable from both Rust and Python.

use polars::export::arrow::array::Utf8ViewArray;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokmat::extractor::{Extractor, MatchMode, ParseOutput};
use tokmat::tel::CompiledPattern;
use tokmat::token_model::TokenModel;
use tokmat::tokenizer::{split_input_tokens_with, tokenize_with_model};
use tokmat::word_definition::configure_word_definition;

static CONTEXT_CACHE: LazyLock<Mutex<HashMap<String, Arc<ModelContext>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct ModelContext {
    model: TokenModel,
    extractor: Extractor,
    features: ModelFeatures,
    type_codec: CompactValueCodec,
    class_codec: CompactValueCodec,
    type_enum_values: Vec<String>,
    class_enum_values: Vec<String>,
    word_definition: WordDefinition,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelFeatures {
    has_postalcode: bool,
    has_num: bool,
    has_alpha: bool,
    has_num_extended: bool,
    has_alpha_extended: bool,
    has_alpha_num: bool,
    has_alpha_num_extended: bool,
}

impl ModelFeatures {
    fn from_model(model: &TokenModel) -> Self {
        let available = model.available_names();
        Self {
            has_postalcode: available.contains("POSTALCODE"),
            has_num: available.contains("NUM"),
            has_alpha: available.contains("ALPHA"),
            has_num_extended: available.contains("NUM_EXTENDED"),
            has_alpha_extended: available.contains("ALPHA_EXTENDED"),
            has_alpha_num: available.contains("ALPHA_NUM"),
            has_alpha_num_extended: available.contains("ALPHA_NUM_EXTENDED"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct CompactExtractProfile {
    rows: usize,
    token_view_ns: Duration,
    class_id_decode_ns: Duration,
    raw_join_ns: Duration,
    parse_ns: Duration,
}

#[derive(Debug)]
struct ChunkExtractOutput {
    field_values: Vec<Vec<Option<String>>>,
    complements: Vec<Option<String>>,
    compact_profile: CompactExtractProfile,
}

const RAW_TOKEN_SENTINEL: u8 = u8::MAX;

#[derive(Debug)]
struct CompactValueCodec {
    values_by_id: Vec<String>,
    ids_by_value: HashMap<String, u8>,
}

impl CompactValueCodec {
    fn new(values: impl IntoIterator<Item = String>, label: &str) -> PolarsResult<Self> {
        let mut values_by_id = values.into_iter().collect::<Vec<_>>();
        values_by_id.sort();
        values_by_id.dedup();

        if values_by_id.len() >= usize::from(RAW_TOKEN_SENTINEL) {
            polars_bail!(
                ComputeError:
                "{} vocabulary has {} entries; exceeds UInt8 compact encoding capacity",
                label,
                values_by_id.len()
            );
        }

        let ids_by_value = values_by_id
            .iter()
            .enumerate()
            .map(|(index, value)| {
                (
                    value.clone(),
                    u8::try_from(index).expect("codec ids are bounded to u8 by construction"),
                )
            })
            .collect();

        Ok(Self {
            values_by_id,
            ids_by_value,
        })
    }

    fn encode_known_or_raw(&self, value: &str) -> u8 {
        self.ids_by_value
            .get(value)
            .copied()
            .unwrap_or(RAW_TOKEN_SENTINEL)
    }

    fn decode_or_fallback_ref<'a>(&'a self, id: u8, raw_token: &'a str) -> Cow<'a, str> {
        if id == RAW_TOKEN_SENTINEL {
            Cow::Borrowed(raw_token)
        } else {
            match self.values_by_id.get(id as usize) {
                Some(value) => Cow::Borrowed(value),
                None => Cow::Borrowed(raw_token),
            }
        }
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy)]
struct TokenizeLayout {
    include_raw_value: bool,
    include_types: bool,
    include_classes: bool,
    include_type_ids: bool,
    include_class_ids: bool,
    token_output: StringListOutput,
    type_output: StringListOutput,
    class_output: StringListOutput,
}

impl TokenizeLayout {
    fn needs_type_values(self) -> bool {
        self.include_types || self.include_type_ids
    }
}

#[allow(clippy::struct_field_names)]
#[derive(Debug)]
struct TokenizedColumns {
    token_values: Vec<Option<Vec<String>>>,
    type_values: Option<Vec<Option<Vec<String>>>>,
    class_values: Option<Vec<Option<Vec<String>>>>,
    type_id_values: Option<Vec<Option<Vec<u8>>>>,
    class_id_values: Option<Vec<Option<Vec<u8>>>>,
}

#[derive(Debug)]
struct TokenizedRow {
    tokens: Vec<String>,
    types: Option<Vec<String>>,
    classes: Option<Vec<String>>,
    type_ids: Option<Vec<u8>>,
    class_ids: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum StringListOutput {
    #[default]
    String,
    Categorical,
    Enum,
}

/// Rust-facing entry point for tokmat-backed Polars operations.
#[derive(Clone)]
pub struct TokmatPolars {
    context: Arc<ModelContext>,
}

impl TokmatPolars {
    /// Load or reuse a cached tokmat model from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the model path cannot be loaded into a valid `tokmat` model.
    pub fn from_model_path(model_path: impl AsRef<Path>) -> PolarsResult<Self> {
        let model_path = model_path.as_ref().to_string_lossy().into_owned();
        let context = get_or_load_context(&model_path)?;
        Ok(Self { context })
    }

    /// Tokenize a UTF-8 series into a struct series containing raw value, tokens,
    /// token types, and token classes.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is not a UTF-8 series or if tokenization fails.
    pub fn tokenize_series(&self, input: &Series) -> PolarsResult<Series> {
        tokenize_series_with_context(
            input,
            &self.context,
            TokenizeLayout {
                include_raw_value: true,
                include_types: true,
                include_classes: true,
                include_type_ids: false,
                include_class_ids: false,
                token_output: StringListOutput::String,
                type_output: StringListOutput::String,
                class_output: StringListOutput::String,
            },
        )
    }

    /// Extract TEL captures from either a UTF-8 series or a tokenized struct series.
    ///
    /// # Errors
    ///
    /// Returns an error if the input dtype is unsupported, the TEL pattern is invalid,
    /// or extraction fails for the configured model.
    pub fn extract_series(&self, input: &Series, pattern: &str) -> PolarsResult<Series> {
        self.extract_series_with_mode(input, pattern, MatchMode::Whole)
    }

    /// Extract TEL captures using the supplied match mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the input dtype is unsupported, the TEL pattern is invalid,
    /// or extraction fails for the configured model.
    pub fn extract_series_with_mode(
        &self,
        input: &Series,
        pattern: &str,
        mode: MatchMode,
    ) -> PolarsResult<Series> {
        extract_series_with_context(input, &self.context, pattern, mode)
    }

    /// Return the TEL capture field names the extractor will emit for a pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the TEL pattern cannot be compiled.
    pub fn capture_field_names(&self, pattern: &str) -> PolarsResult<Vec<String>> {
        let _ = &self.context;
        capture_field_names_from_pattern(pattern)
    }

    /// The active word definition for this model — the character class (without
    /// brackets, e.g. `\w\-'`) that delimits tokens. Loaded from the model's
    /// `WORDDEFINITION.param`, or the default if absent.
    #[must_use]
    pub fn word_definition(&self) -> &str {
        self.context.word_definition.chars()
    }
}

// Re-export so Rust callers can construct/inspect word definitions without
// depending on `tokmat` directly.
pub use tokmat::word_definition::WordDefinition;

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Deserialize)]
struct TokenizeKwargs {
    model_path: String,
    #[serde(default = "default_true")]
    include_raw_value: bool,
    #[serde(default = "default_true")]
    include_types: bool,
    #[serde(default = "default_true")]
    include_classes: bool,
    #[serde(default)]
    include_type_ids: bool,
    #[serde(default)]
    include_class_ids: bool,
    #[serde(default)]
    token_output: StringListOutput,
    #[serde(default)]
    type_output: StringListOutput,
    #[serde(default)]
    class_output: StringListOutput,
}

const fn default_true() -> bool {
    true
}

impl TokenizeKwargs {
    fn layout(&self) -> PolarsResult<TokenizeLayout> {
        if self.token_output == StringListOutput::Enum
            || self.type_output == StringListOutput::Enum
            || self.class_output == StringListOutput::Enum
        {
            polars_bail!(
                InvalidOperation:
                "enum list output is not supported in tokmat-polars; use 'string' or 'categorical'"
            );
        }
        Ok(TokenizeLayout {
            include_raw_value: self.include_raw_value,
            include_types: self.include_types,
            include_classes: self.include_classes,
            include_type_ids: self.include_type_ids,
            include_class_ids: self.include_class_ids,
            token_output: self.token_output,
            type_output: self.type_output,
            class_output: self.class_output,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum MatchModeKwarg {
    #[default]
    Whole,
    Start,
    End,
    Any,
}

impl From<MatchModeKwarg> for MatchMode {
    fn from(mode: MatchModeKwarg) -> Self {
        match mode {
            MatchModeKwarg::Whole => Self::Whole,
            MatchModeKwarg::Start => Self::Start,
            MatchModeKwarg::End => Self::End,
            MatchModeKwarg::Any => Self::Any,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ExtractKwargs {
    model_path: String,
    pattern: String,
    #[serde(default)]
    mode: MatchModeKwarg,
}

#[derive(Debug, Clone, Deserialize)]
struct EncodeClassIdsKwargs {
    model_path: String,
}

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func_with_kwargs=tokenize_output_type)]
fn tokenize_expr(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    tokenize_expr_impl(inputs, &kwargs)
}

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func_with_kwargs=extract_output_type)]
fn extract_expr(inputs: &[Series], kwargs: ExtractKwargs) -> PolarsResult<Series> {
    extract_expr_impl(inputs, kwargs)
}

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func_with_kwargs=encode_class_ids_output_type)]
fn encode_class_ids_expr(inputs: &[Series], kwargs: EncodeClassIdsKwargs) -> PolarsResult<Series> {
    encode_class_ids_expr_impl(inputs, &kwargs)
}

/// Return the word-definition character class for a model (from its
/// `WORDDEFINITION.param`, or the default `\w\-'` if absent).
#[pyfunction]
fn model_word_definition(model_path: &str) -> PyResult<String> {
    let context = get_or_load_context(model_path)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(context.word_definition.chars().to_string())
}

/// Read the process-global word definition (the character class string).
///
/// Note: model tokenization uses each model's own boundary, not this global;
/// this reflects the last value set via [`set_word_definition`] (or the default).
#[pyfunction]
fn get_word_definition() -> String {
    tokmat::word_definition::word_definition()
        .chars()
        .to_string()
}

/// Set the process-global word definition (character class, with or without
/// surrounding brackets, e.g. `\w\-'.` or `[\w\-'.]`).
///
/// This affects the global-reading TEL/extractor path. It is process-wide and
/// not thread-scoped; prefer driving word definitions via a model's
/// `WORDDEFINITION.param`.
#[pyfunction]
fn set_word_definition(chars: &str) {
    configure_word_definition(WordDefinition::new(chars));
}

// The compiled extension is a private submodule (`tokmat_polars._tokmat_polars`)
// of the `tokmat_polars` Python package. The package's `__init__.py` provides the
// user-facing API (which registers the plugin expressions with
// `is_elementwise=True`). Polars loads the plugin functions from this compiled
// module via its path.
#[pymodule]
fn _tokmat_polars(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(model_word_definition, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(get_word_definition, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(set_word_definition, module)?)?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn tokenize_output_type(input_fields: &[Field], kwargs: TokenizeKwargs) -> PolarsResult<Field> {
    let output_name = output_field_name(input_fields, "tokenized");
    let context = get_or_load_context(&kwargs.model_path)?;
    let layout = kwargs.layout()?;
    Ok(Field::new(
        output_name,
        DataType::Struct(tokenize_fields(&context, layout)?),
    ))
}

#[allow(clippy::needless_pass_by_value)]
fn extract_output_type(input_fields: &[Field], kwargs: ExtractKwargs) -> PolarsResult<Field> {
    let output_name = output_field_name(input_fields, "extracted");
    let capture_names = capture_field_names_from_pattern(&kwargs.pattern)?;
    Ok(Field::new(
        output_name,
        DataType::Struct(extract_fields(&capture_names)),
    ))
}

#[allow(clippy::needless_pass_by_value, clippy::unnecessary_wraps)]
fn encode_class_ids_output_type(
    input_fields: &[Field],
    _kwargs: EncodeClassIdsKwargs,
) -> PolarsResult<Field> {
    let output_name = output_field_name(input_fields, "class_ids");
    Ok(Field::new(
        output_name,
        DataType::List(Box::new(DataType::UInt8)),
    ))
}

fn tokenize_expr_impl(inputs: &[Series], kwargs: &TokenizeKwargs) -> PolarsResult<Series> {
    let input = single_input(inputs, "tokenize_expr")?;
    let context = get_or_load_context(&kwargs.model_path)?;
    tokenize_series_with_context(input, &context, kwargs.layout()?)
}

fn extract_expr_impl(inputs: &[Series], kwargs: ExtractKwargs) -> PolarsResult<Series> {
    let input = single_input(inputs, "extract_expr")?;
    let context = get_or_load_context(&kwargs.model_path)?;
    extract_series_with_context(input, &context, &kwargs.pattern, kwargs.mode.into())
}

fn encode_class_ids_expr_impl(
    inputs: &[Series],
    kwargs: &EncodeClassIdsKwargs,
) -> PolarsResult<Series> {
    let input = single_input(inputs, "encode_class_ids_expr")?;
    let context = get_or_load_context(&kwargs.model_path)?;
    let mut rows = Vec::with_capacity(input.len());
    for row in input.list()? {
        match row {
            Some(classes) => {
                let class_values = list_series_to_str_views(&classes)?;
                rows.push(Some(
                    class_values
                        .into_iter()
                        .map(|value| context.class_codec.encode_known_or_raw(value))
                        .collect(),
                ));
            }
            None => rows.push(None),
        }
    }
    Ok(build_u8_list_series(input.name(), rows))
}

fn tokenize_series_with_context(
    input: &Series,
    context: &ModelContext,
    layout: TokenizeLayout,
) -> PolarsResult<Series> {
    if should_parallelize_tokenize(input.len()) {
        return tokenize_series_parallel(input, context, layout);
    }
    tokenize_series_serial(input, context, layout)
}

fn tokenize_series_serial(
    input: &Series,
    context: &ModelContext,
    layout: TokenizeLayout,
) -> PolarsResult<Series> {
    if can_tokenize_direct(layout) {
        return tokenize_series_with_context_direct(input, context, layout);
    }
    tokenize_series_with_context_staged(input, context, layout)
}

/// Tokenize row-range chunks in parallel, then concatenate. Rows are
/// independent, so each chunk runs the full serial path over a zero-copy slice
/// of the input and the resulting struct chunks are appended in order.
fn tokenize_series_parallel(
    input: &Series,
    context: &ModelContext,
    layout: TokenizeLayout,
) -> PolarsResult<Series> {
    tokenize_series_chunked(input, context, layout, parallel_chunk_size(input.len()))
}

fn tokenize_series_chunked(
    input: &Series,
    context: &ModelContext,
    layout: TokenizeLayout,
    chunk_size: usize,
) -> PolarsResult<Series> {
    let row_count = input.len();
    let chunk_series = (0..row_count)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(row_count)))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(start, end)| {
            let slice = input.slice(
                i64::try_from(start).expect("row index fits in i64"),
                end - start,
            );
            tokenize_series_serial(&slice, context, layout)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut chunks = chunk_series.into_iter();
    let mut combined = chunks
        .next()
        .expect("row_count >= PARALLEL_ROW_THRESHOLD guarantees at least one chunk");
    for chunk in chunks {
        combined.append(&chunk)?;
    }
    Ok(combined)
}

fn can_tokenize_direct(layout: TokenizeLayout) -> bool {
    layout.token_output == StringListOutput::String
        && (!layout.include_types || layout.type_output == StringListOutput::String)
        && (!layout.include_classes || layout.class_output == StringListOutput::String)
}

fn tokenize_series_with_context_staged(
    input: &Series,
    context: &ModelContext,
    layout: TokenizeLayout,
) -> PolarsResult<Series> {
    let row_count = input.len();
    let mut columns = TokenizedColumns {
        token_values: Vec::with_capacity(row_count),
        type_values: layout
            .needs_type_values()
            .then(|| Vec::with_capacity(row_count)),
        class_values: layout
            .include_classes
            .then(|| Vec::with_capacity(row_count)),
        type_id_values: layout
            .include_type_ids
            .then(|| Vec::with_capacity(row_count)),
        class_id_values: layout
            .include_class_ids
            .then(|| Vec::with_capacity(row_count)),
    };

    for value in input.str()? {
        if let Some(raw_value) = value {
            let tokenized = tokenize_row(raw_value, context, layout);
            columns.token_values.push(Some(tokenized.tokens));
            if let Some(type_values) = columns.type_values.as_mut() {
                type_values.push(tokenized.types);
            }
            if let Some(class_values) = columns.class_values.as_mut() {
                class_values.push(tokenized.classes);
            }
            if let Some(type_id_values) = columns.type_id_values.as_mut() {
                type_id_values.push(tokenized.type_ids);
            }
            if let Some(class_id_values) = columns.class_id_values.as_mut() {
                class_id_values.push(tokenized.class_ids);
            }
        } else {
            columns.token_values.push(None);
            if let Some(type_values) = columns.type_values.as_mut() {
                type_values.push(None);
            }
            if let Some(class_values) = columns.class_values.as_mut() {
                class_values.push(None);
            }
            if let Some(type_id_values) = columns.type_id_values.as_mut() {
                type_id_values.push(None);
            }
            if let Some(class_id_values) = columns.class_id_values.as_mut() {
                class_id_values.push(None);
            }
        }
    }

    build_tokenized_struct_series(input, columns, context, layout)
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn tokenize_series_with_context_direct(
    input: &Series,
    context: &ModelContext,
    layout: TokenizeLayout,
) -> PolarsResult<Series> {
    let row_count = input.len();
    let input_total_bytes = input
        .str()?
        .into_iter()
        .flatten()
        .map(str::len)
        .sum::<usize>();

    let mut token_builder =
        ListStringChunkedBuilder::new("tokens".into(), row_count, input_total_bytes);
    let mut type_builder = layout
        .include_types
        .then(|| ListStringChunkedBuilder::new("types".into(), row_count, input_total_bytes));
    let mut class_builder = layout
        .include_classes
        .then(|| ListStringChunkedBuilder::new("classes".into(), row_count, input_total_bytes));
    let mut type_id_builder = layout.include_type_ids.then(|| {
        ListPrimitiveChunkedBuilder::<UInt8Type>::new(
            "type_ids".into(),
            row_count,
            row_count * 8,
            DataType::UInt8,
        )
    });
    let mut class_id_builder = layout.include_class_ids.then(|| {
        ListPrimitiveChunkedBuilder::<UInt8Type>::new(
            "class_ids".into(),
            row_count,
            row_count * 8,
            DataType::UInt8,
        )
    });

    for value in input.str()? {
        if let Some(raw_value) = value {
            let tokens = split_input_tokens_with(raw_value, context.model.word_boundary());
            token_builder.append_values_iter(tokens.iter().map(String::as_str));

            let needs_row_types = layout.include_types || layout.include_type_ids;
            let needs_row_classes = layout.include_classes || layout.include_class_ids;

            let mut row_types = layout
                .include_types
                .then(SmallVec::<[Cow<'_, str>; 12]>::new);
            let mut row_classes = layout
                .include_classes
                .then(SmallVec::<[Cow<'_, str>; 12]>::new);
            let mut row_type_ids = layout.include_type_ids.then(SmallVec::<[u8; 12]>::new);
            let mut row_class_ids = layout.include_class_ids.then(SmallVec::<[u8; 12]>::new);

            if needs_row_types || needs_row_classes {
                for token in &tokens {
                    let token_type = classify_token_ref(token, &context.model, context.features);
                    if let Some(values) = row_types.as_mut() {
                        values.push(token_type.clone());
                    }
                    if let Some(values) = row_type_ids.as_mut() {
                        values.push(context.type_codec.encode_known_or_raw(token_type.as_ref()));
                    }

                    let class_value = if token.chars().all(char::is_whitespace) {
                        Cow::Borrowed(token.as_str())
                    } else if let Some(value) = context.model.token_class_lookup().get(token) {
                        Cow::Borrowed(value.as_str())
                    } else {
                        Cow::Owned(token_type.as_ref().to_string())
                    };

                    if let Some(values) = row_classes.as_mut() {
                        values.push(class_value.clone());
                    }
                    if let Some(values) = row_class_ids.as_mut() {
                        values.push(
                            context
                                .class_codec
                                .encode_known_or_raw(class_value.as_ref()),
                        );
                    }
                }
            }

            if let Some(builder) = type_builder.as_mut() {
                if let Some(values) = row_types {
                    builder.append_values_iter(values.iter().map(AsRef::as_ref));
                } else {
                    builder.append_values_iter(std::iter::empty::<&str>());
                }
            }
            if let Some(builder) = class_builder.as_mut() {
                if let Some(values) = row_classes {
                    builder.append_values_iter(values.iter().map(AsRef::as_ref));
                } else {
                    builder.append_values_iter(std::iter::empty::<&str>());
                }
            }
            if let Some(builder) = type_id_builder.as_mut() {
                if let Some(values) = row_type_ids {
                    builder.append_slice(values.as_slice());
                } else {
                    builder.append_slice(&[]);
                }
            }
            if let Some(builder) = class_id_builder.as_mut() {
                if let Some(values) = row_class_ids {
                    builder.append_slice(values.as_slice());
                } else {
                    builder.append_slice(&[]);
                }
            }
        } else {
            token_builder.append_null();
            if let Some(builder) = type_builder.as_mut() {
                builder.append_null();
            }
            if let Some(builder) = class_builder.as_mut() {
                builder.append_null();
            }
            if let Some(builder) = type_id_builder.as_mut() {
                builder.append_null();
            }
            if let Some(builder) = class_id_builder.as_mut() {
                builder.append_null();
            }
        }
    }

    let mut fields = Vec::new();
    if layout.include_raw_value {
        // `raw_value` is byte-for-byte the input column, so reuse the existing
        // Arrow buffers (Arc clone + rename) instead of reconstructing and
        // re-copying the whole column cell by cell.
        let mut raw = input.clone();
        raw.rename("raw_value".into());
        fields.push(raw);
    }
    fields.push(token_builder.finish().into_series());
    if let Some(mut builder) = type_builder {
        fields.push(builder.finish().into_series());
    }
    if let Some(mut builder) = class_builder {
        fields.push(builder.finish().into_series());
    }
    if let Some(mut builder) = type_id_builder {
        fields.push(builder.finish().into_series());
    }
    if let Some(mut builder) = class_id_builder {
        fields.push(builder.finish().into_series());
    }

    Ok(StructChunked::from_series(input.name().clone(), row_count, fields.iter())?.into_series())
}

fn extract_series_with_context(
    input: &Series,
    context: &ModelContext,
    pattern: &str,
    mode: MatchMode,
) -> PolarsResult<Series> {
    // The model's word definition is threaded explicitly through
    // `CompiledPattern::compile_with_word_definition` (see `compile_extract_pattern`)
    // and into the object-plan builder via the per-model `Extractor`, so extraction
    // honors this model's `WORDDEFINITION.param` without touching process-global
    // state. This is what makes concurrent extraction of *different* models (e.g.
    // the polars streaming engine invoking elementwise plugin expressions per morsel
    // across passes) deterministic — the previous `configure_word_definition` write
    // raced when models had differing word definitions.
    let capture_names = capture_field_names_from_pattern(pattern)?;

    match input.dtype() {
        DataType::String => {
            extract_from_string_series(input, context, pattern, mode, &capture_names)
        }
        DataType::Struct(_) => {
            extract_from_tokenized_series(input, context, pattern, mode, &capture_names)
        }
        dtype => {
            polars_bail!(
                InvalidOperation:
                "extract_series expected a String or tokenized Struct column, got {:?}",
                dtype
            )
        }
    }
}

fn get_or_load_context(model_path: &str) -> PolarsResult<Arc<ModelContext>> {
    let mut cache = CONTEXT_CACHE
        .lock()
        .map_err(|error| polars_err!(ComputeError: "context cache poisoned: {}", error))?;

    if let Some(context) = cache.get(model_path) {
        return Ok(Arc::clone(context));
    }

    let model = TokenModel::load(Path::new(model_path)).map_err(|error| {
        polars_err!(
            ComputeError:
            "failed to load tokmat model from '{}': {}",
            model_path,
            error
        )
    })?;
    let extractor = Extractor::new(
        model.token_definitions().clone(),
        model.token_class_list().clone(),
    )
    .with_word_definition(model.word_definition().clone());

    let type_vocab = model
        .token_definitions()
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let class_vocab = model
        .token_definitions()
        .iter()
        .map(|(name, _)| name.clone())
        .chain(model.token_class_lookup().values().cloned())
        .collect::<Vec<_>>();
    let type_codec = CompactValueCodec::new(type_vocab.clone(), "type")?;
    let class_codec = CompactValueCodec::new(class_vocab.clone(), "class")?;
    let type_enum_values = enum_categories(type_vocab.into_iter().chain([" ".to_string()]));
    let class_enum_values = enum_categories(class_vocab.into_iter().chain([" ".to_string()]));

    let features = ModelFeatures::from_model(&model);
    let word_definition = model.word_definition().clone();
    let context = Arc::new(ModelContext {
        model,
        extractor,
        features,
        type_codec,
        class_codec,
        type_enum_values,
        class_enum_values,
        word_definition,
    });
    cache.insert(model_path.to_string(), Arc::clone(&context));
    Ok(context)
}

fn single_input<'a>(inputs: &'a [Series], function_name: &str) -> PolarsResult<&'a Series> {
    match inputs {
        [input] => Ok(input),
        _ => polars_bail!(
            InvalidOperation:
            "{} expected exactly one input column, got {}",
            function_name,
            inputs.len()
        ),
    }
}

fn output_field_name(input_fields: &[Field], fallback: &str) -> PlSmallStr {
    input_fields
        .first()
        .map_or_else(|| fallback.into(), |field| field.name().clone())
}

fn enum_categories(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut categories = values.into_iter().collect::<Vec<_>>();
    categories.sort();
    categories.dedup();
    categories
}

fn enum_dtype(values: &[String]) -> DataType {
    let categories = Utf8ViewArray::from_slice(
        values
            .iter()
            .map(|value| Some(value.as_str()))
            .collect::<Vec<_>>(),
    );
    create_enum_dtype(categories)
}

fn list_output_dtype(
    output: StringListOutput,
    enum_values: Option<&[String]>,
) -> PolarsResult<DataType> {
    match output {
        StringListOutput::String => Ok(DataType::String),
        StringListOutput::Categorical => {
            Ok(DataType::Categorical(None, CategoricalOrdering::default()))
        }
        StringListOutput::Enum => enum_values
            .map(enum_dtype)
            .ok_or_else(|| polars_err!(InvalidOperation: "enum output requires fixed categories")),
    }
}

fn cast_string_list_series(
    series: Series,
    output: StringListOutput,
    enum_values: Option<&[String]>,
) -> PolarsResult<Series> {
    let target_dtype = DataType::List(Box::new(list_output_dtype(output, enum_values)?));
    if series.dtype() == &target_dtype {
        Ok(series)
    } else {
        series.cast(&target_dtype)
    }
}

fn build_output_string_list_series(
    name: &str,
    rows: Vec<Option<Vec<String>>>,
    output: StringListOutput,
    enum_values: Option<&[String]>,
) -> PolarsResult<Series> {
    match output {
        StringListOutput::String => Ok(build_string_list_series(name, rows)),
        StringListOutput::Categorical => {
            cast_string_list_series(build_string_list_series(name, rows), output, enum_values)
        }
        StringListOutput::Enum => build_enum_list_series(
            name,
            rows,
            enum_values.ok_or_else(
                || polars_err!(InvalidOperation: "enum output requires fixed categories"),
            )?,
        ),
    }
}

fn tokenize_fields(context: &ModelContext, layout: TokenizeLayout) -> PolarsResult<Vec<Field>> {
    let mut fields = Vec::with_capacity(6);
    if layout.include_raw_value {
        fields.push(Field::new("raw_value".into(), DataType::String));
    }
    fields.push(Field::new(
        "tokens".into(),
        DataType::List(Box::new(list_output_dtype(layout.token_output, None)?)),
    ));
    if layout.include_types {
        fields.push(Field::new(
            "types".into(),
            DataType::List(Box::new(list_output_dtype(
                layout.type_output,
                Some(&context.type_enum_values),
            )?)),
        ));
    }
    if layout.include_classes {
        fields.push(Field::new(
            "classes".into(),
            DataType::List(Box::new(list_output_dtype(
                layout.class_output,
                Some(&context.class_enum_values),
            )?)),
        ));
    }
    if layout.include_type_ids {
        fields.push(Field::new(
            "type_ids".into(),
            DataType::List(Box::new(DataType::UInt8)),
        ));
    }
    if layout.include_class_ids {
        fields.push(Field::new(
            "class_ids".into(),
            DataType::List(Box::new(DataType::UInt8)),
        ));
    }
    Ok(fields)
}

fn extract_fields(capture_names: &[String]) -> Vec<Field> {
    let mut fields = capture_names
        .iter()
        .map(|name| Field::new(name.clone().into(), DataType::String))
        .collect::<Vec<_>>();
    fields.push(Field::new("complement".into(), DataType::String));
    fields
}

fn capture_field_names_from_pattern(pattern: &str) -> PolarsResult<Vec<String>> {
    let compiled = CompiledPattern::compile(pattern).map_err(|error| {
        polars_err!(
            ComputeError:
            "failed to compile TEL pattern '{}': {}",
            pattern,
            error
        )
    })?;

    let mut seen = HashSet::new();
    let mut fields = Vec::new();
    for token_info in compiled.token_info() {
        let Some(name) = token_info.var_name.as_ref() else {
            continue;
        };
        if !(token_info.is_capturing_group() || token_info.is_vanishing_group()) {
            continue;
        }
        if seen.insert(name.clone()) {
            fields.push(name.clone());
        }
    }

    Ok(fields)
}

fn extract_from_string_series(
    input: &Series,
    context: &ModelContext,
    pattern: &str,
    mode: MatchMode,
    capture_names: &[String],
) -> PolarsResult<Series> {
    let row_count = input.len();
    let compiled_pattern = compile_extract_pattern(pattern, &context.word_definition)?;
    let extractor = &context.extractor;
    let mut builders = ExtractColumnBuilders::new(capture_names, row_count);

    for raw_value in input.str()? {
        match raw_value {
            Some(raw_value) => {
                let tokenized = tokenize_with_model(raw_value, &context.model);
                let parsed = parse_from_tokenized_parts(
                    extractor,
                    raw_value,
                    &tokenized.tokens,
                    &tokenized.classes,
                    &compiled_pattern,
                    mode,
                )?;
                builders.push(capture_names, Some(parsed));
            }
            None => builders.push(capture_names, None),
        }
    }

    builders.finish(input.name().clone(), row_count)
}

/// Ensure a tokenized struct list-field is `List(String)`.
///
/// Token/class columns can be `List(Categorical)` or `List(Enum)` (categorical
/// tokenize output), but the borrowing `&str`-view extraction path requires
/// `List(String)`. Cast such a column to `List(String)` once (rather than per
/// row); a `List(String)` column is returned as a cheap clone (no data copy).
fn ensure_string_list_field(field: &Series) -> PolarsResult<Series> {
    match field.dtype() {
        DataType::List(inner)
            if matches!(
                inner.as_ref(),
                DataType::Categorical(_, _) | DataType::Enum(_, _)
            ) =>
        {
            field.cast(&DataType::List(Box::new(DataType::String)))
        }
        _ => Ok(field.clone()),
    }
}

#[allow(clippy::too_many_lines)]
fn extract_from_tokenized_series(
    input: &Series,
    context: &ModelContext,
    pattern: &str,
    mode: MatchMode,
    capture_names: &[String],
) -> PolarsResult<Series> {
    let struct_chunked = input.struct_()?;
    let fields = struct_chunked.fields_as_series();
    let field_map = fields
        .into_iter()
        .map(|field| (field.name().to_string(), field))
        .collect::<HashMap<_, _>>();

    let raw_field = field_map.get("raw_value");
    let tokens_field = field_map.get("tokens").ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "tokenized struct is missing required 'tokens' field"
        )
    })?;
    let classes_field = field_map.get("classes");
    let class_ids_field = field_map.get("class_ids");
    if classes_field.is_none() && class_ids_field.is_none() {
        polars_bail!(
            InvalidOperation:
            "tokenized struct is missing required 'classes' or 'class_ids' field"
        );
    }

    // Token/class columns may be categorical/enum lists (from categorical
    // tokenize output); the `&str`-view path needs `List(String)`. Cast once here
    // so both the serial and parallel paths below see String lists.
    let tokens_field = ensure_string_list_field(tokens_field)?;
    let classes_field = classes_field.map(ensure_string_list_field).transpose()?;

    if should_parallelize_extract(input.len()) {
        return extract_from_tokenized_series_parallel(
            input,
            context,
            pattern,
            mode,
            capture_names,
            raw_field,
            &tokens_field,
            classes_field.as_ref(),
            class_ids_field,
        );
    }

    let mut raw_iter = raw_field
        .map(|field| field.str().map(IntoIterator::into_iter))
        .transpose()?;
    let mut token_iter = tokens_field.list()?.into_iter();
    let mut class_iter = classes_field
        .as_ref()
        .map(|field| field.list().map(IntoIterator::into_iter))
        .transpose()?;
    let mut class_id_iter = class_ids_field
        .map(|field| field.list().map(IntoIterator::into_iter))
        .transpose()?;
    let row_count = input.len();
    let compiled_pattern = compile_extract_pattern(pattern, &context.word_definition)?;
    let extractor = &context.extractor;
    let mut builders = ExtractColumnBuilders::new(capture_names, row_count);
    let profiling = profile_enabled();
    let mut compact_profile = profiling.then(CompactExtractProfile::default);

    for index in 0..input.len() {
        let raw_value = raw_iter.as_mut().and_then(Iterator::next).flatten();
        let tokens = token_iter.next();
        let classes = class_iter.as_mut().and_then(Iterator::next);
        let class_ids = class_id_iter.as_mut().and_then(Iterator::next);
        match (tokens, classes, class_ids) {
            (Some(Some(tokens)), Some(Some(classes)), _) => {
                let token_values = list_series_to_str_views(&tokens)?;
                let class_values = list_series_to_str_views(&classes)?;
                let raw_value_buf = raw_value.map_or_else(
                    || Cow::Owned(join_string_values(&token_values)),
                    Cow::Borrowed,
                );
                let parsed = parse_from_tokenized_parts(
                    extractor,
                    raw_value_buf,
                    &token_values,
                    &class_values,
                    &compiled_pattern,
                    mode,
                )?;
                builders.push(capture_names, Some(parsed));
            }
            (Some(Some(tokens)), _, Some(Some(class_ids))) => {
                let token_view_start = profiling.then(Instant::now);
                let token_values = list_series_to_str_views(&tokens)?;
                let token_view_elapsed = elapsed_since(token_view_start);
                let decode_start = profiling.then(Instant::now);
                let class_values = list_series_to_u8(&class_ids)?
                    .into_iter()
                    .zip(token_values.iter().copied())
                    .map(|(class_id, token)| {
                        context.class_codec.decode_or_fallback_ref(class_id, token)
                    })
                    .collect::<Vec<_>>();
                let decode_elapsed = elapsed_since(decode_start);
                let raw_join_start = profiling
                    .then_some(raw_value.is_none())
                    .filter(|should_join| *should_join)
                    .map(|_| Instant::now());
                let raw_value_buf = raw_value.map_or_else(
                    || Cow::Owned(join_string_values(&token_values)),
                    Cow::Borrowed,
                );
                let raw_join_elapsed = elapsed_since(raw_join_start);
                let parse_start = profiling.then(Instant::now);
                let parsed = parse_from_tokenized_parts(
                    extractor,
                    raw_value_buf,
                    &token_values,
                    &class_values,
                    &compiled_pattern,
                    mode,
                )?;
                let parse_elapsed = elapsed_since(parse_start);
                if let Some(profile) = compact_profile.as_mut() {
                    profile.rows += 1;
                    profile.token_view_ns += token_view_elapsed;
                    profile.class_id_decode_ns += decode_elapsed;
                    profile.raw_join_ns += raw_join_elapsed;
                    profile.parse_ns += parse_elapsed;
                }
                builders.push(capture_names, Some(parsed));
            }
            (Some(None), Some(None) | None, Some(None) | None) if raw_value.is_none() => {
                builders.push(capture_names, None);
            }
            _ => {
                polars_bail!(
                    InvalidOperation:
                    "tokenized struct row {} has inconsistent nullability across fields",
                    index
                )
            }
        }
    }

    if let Some(profile) = compact_profile {
        if let Ok(stats) = context.extractor.stats() {
            eprintln!(
                "TOKMAT_PROFILE compact rows={} token_view_ns={} class_id_decode_ns={} raw_join_ns={} parse_ns={} tokmat_profiled_rows={} tokmat_total_ns={} tokmat_class_join_ns={} tokmat_class_regex_ns={} tokmat_offset_work_ns={} tokmat_object_join_ns={} tokmat_direct_execution_ns={} tokmat_fallback_regex_ns={}",
                profile.rows,
                profile.token_view_ns.as_nanos(),
                profile.class_id_decode_ns.as_nanos(),
                profile.raw_join_ns.as_nanos(),
                profile.parse_ns.as_nanos(),
                stats.profiled_rows,
                stats.profile_total_ns,
                stats.profile_class_join_ns,
                stats.profile_class_regex_ns,
                stats.profile_offset_work_ns,
                stats.profile_object_join_ns,
                stats.profile_direct_execution_ns,
                stats.profile_fallback_regex_ns,
            );
        }
    }

    builders.finish(input.name().clone(), row_count)
}

#[allow(clippy::too_many_arguments)]
fn extract_from_tokenized_series_parallel(
    input: &Series,
    context: &ModelContext,
    pattern: &str,
    mode: MatchMode,
    capture_names: &[String],
    raw_field: Option<&Series>,
    tokens_field: &Series,
    classes_field: Option<&Series>,
    class_ids_field: Option<&Series>,
) -> PolarsResult<Series> {
    let row_count = input.len();
    let chunk_size = parallel_chunk_size(row_count);
    let profiling = profile_enabled();
    let chunk_ranges = (0..row_count)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(row_count)))
        .collect::<Vec<_>>();

    let raw_series = raw_field.cloned();
    let tokens_series = tokens_field.clone();
    let classes_series = classes_field.cloned();
    let class_ids_series = class_ids_field.cloned();

    let chunk_results = chunk_ranges
        .into_par_iter()
        .map(|(start, end)| {
            process_extract_chunk(
                raw_series.as_ref(),
                &tokens_series,
                classes_series.as_ref(),
                class_ids_series.as_ref(),
                context,
                pattern,
                mode,
                capture_names,
                profiling,
                start,
                end,
            )
        })
        .collect::<Vec<_>>();

    let mut field_values = capture_names
        .iter()
        .map(|_| Vec::with_capacity(row_count))
        .collect::<Vec<_>>();
    let mut complements = Vec::with_capacity(row_count);
    let mut merged_profile = CompactExtractProfile::default();

    for chunk_result in chunk_results {
        let chunk = chunk_result?;
        for (index, values) in chunk.field_values.into_iter().enumerate() {
            field_values[index].extend(values);
        }
        complements.extend(chunk.complements);
        merged_profile.rows += chunk.compact_profile.rows;
        merged_profile.token_view_ns += chunk.compact_profile.token_view_ns;
        merged_profile.class_id_decode_ns += chunk.compact_profile.class_id_decode_ns;
        merged_profile.raw_join_ns += chunk.compact_profile.raw_join_ns;
        merged_profile.parse_ns += chunk.compact_profile.parse_ns;
    }

    if profiling {
        if let Ok(stats) = context.extractor.stats() {
            eprintln!(
                "TOKMAT_PROFILE compact rows={} token_view_ns={} class_id_decode_ns={} raw_join_ns={} parse_ns={} tokmat_profiled_rows={} tokmat_total_ns={} tokmat_class_join_ns={} tokmat_class_regex_ns={} tokmat_offset_work_ns={} tokmat_object_join_ns={} tokmat_direct_execution_ns={} tokmat_fallback_regex_ns={}",
                merged_profile.rows,
                merged_profile.token_view_ns.as_nanos(),
                merged_profile.class_id_decode_ns.as_nanos(),
                merged_profile.raw_join_ns.as_nanos(),
                merged_profile.parse_ns.as_nanos(),
                stats.profiled_rows,
                stats.profile_total_ns,
                stats.profile_class_join_ns,
                stats.profile_class_regex_ns,
                stats.profile_offset_work_ns,
                stats.profile_object_join_ns,
                stats.profile_direct_execution_ns,
                stats.profile_fallback_regex_ns,
            );
        }
    }

    let named_field_values = capture_names
        .iter()
        .cloned()
        .zip(field_values)
        .collect::<Vec<_>>();
    build_extract_struct_series(input.name().clone(), named_field_values, &complements)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn process_extract_chunk(
    raw_series: Option<&Series>,
    tokens_series: &Series,
    classes_series: Option<&Series>,
    class_ids_series: Option<&Series>,
    context: &ModelContext,
    pattern: &str,
    mode: MatchMode,
    capture_names: &[String],
    profiling: bool,
    start: usize,
    end: usize,
) -> PolarsResult<ChunkExtractOutput> {
    let raw_utf8 = raw_series.map(Series::str).transpose()?;
    let token_lists = tokens_series.list()?;
    let class_lists = classes_series.map(Series::list).transpose()?;
    let class_id_lists = class_ids_series.map(Series::list).transpose()?;

    let mut field_values = capture_names
        .iter()
        .map(|_| Vec::with_capacity(end - start))
        .collect::<Vec<_>>();
    let mut complements = Vec::with_capacity(end - start);
    let mut compact_profile = CompactExtractProfile::default();
    let mut class_id_values = Vec::new();
    let compiled_pattern = compile_extract_pattern(pattern, &context.word_definition)?;
    // Per-chunk extractor: its own caches, so rayon workers never contend on the
    // shared extractor's Mutex-guarded plan/regex caches (the root cause of the
    // parallel-extract regression). See `fresh_chunk_extractor`.
    let chunk_extractor = fresh_chunk_extractor(context);
    let extractor = &chunk_extractor;

    for index in start..end {
        let raw_value = raw_utf8.as_ref().and_then(|values| values.get(index));
        let tokens = token_lists.get_as_series(index);
        let classes = class_lists
            .as_ref()
            .and_then(|values| values.get_as_series(index));
        let class_ids = class_id_lists
            .as_ref()
            .and_then(|values| values.get_as_series(index));

        match (tokens, classes, class_ids) {
            (Some(tokens), Some(classes), _) => {
                let token_values = list_series_to_str_views(&tokens)?;
                let class_values = list_series_to_str_views(&classes)?;
                let raw_value_buf = raw_value.map_or_else(
                    || Cow::Owned(join_string_values(&token_values)),
                    Cow::Borrowed,
                );
                let parsed = parse_from_tokenized_parts(
                    extractor,
                    raw_value_buf,
                    &token_values,
                    &class_values,
                    &compiled_pattern,
                    mode,
                )?;
                push_parse_output_by_index(
                    &mut field_values,
                    &mut complements,
                    capture_names,
                    Some(parsed),
                );
            }
            (Some(tokens), _, Some(class_ids)) => {
                let token_view_start = profiling.then(Instant::now);
                let token_values = list_series_to_str_views(&tokens)?;
                let token_view_elapsed = elapsed_since(token_view_start);

                let decode_start = profiling.then(Instant::now);
                fill_series_u8(&class_ids, &mut class_id_values)?;
                let class_values = class_id_values
                    .iter()
                    .zip(token_values.iter().copied())
                    .map(|(class_id, token)| {
                        context.class_codec.decode_or_fallback_ref(*class_id, token)
                    })
                    .collect::<Vec<_>>();
                let decode_elapsed = elapsed_since(decode_start);

                let raw_join_start = profiling
                    .then_some(raw_value.is_none())
                    .filter(|should_join| *should_join)
                    .map(|_| Instant::now());
                let raw_value_buf = raw_value.map_or_else(
                    || Cow::Owned(join_string_values(&token_values)),
                    Cow::Borrowed,
                );
                let raw_join_elapsed = elapsed_since(raw_join_start);

                let parse_start = profiling.then(Instant::now);
                let parsed = parse_from_tokenized_parts(
                    extractor,
                    raw_value_buf,
                    &token_values,
                    &class_values,
                    &compiled_pattern,
                    mode,
                )?;
                let parse_elapsed = elapsed_since(parse_start);

                compact_profile.rows += 1;
                compact_profile.token_view_ns += token_view_elapsed;
                compact_profile.class_id_decode_ns += decode_elapsed;
                compact_profile.raw_join_ns += raw_join_elapsed;
                compact_profile.parse_ns += parse_elapsed;

                push_parse_output_by_index(
                    &mut field_values,
                    &mut complements,
                    capture_names,
                    Some(parsed),
                );
            }
            (None, None, None) if raw_value.is_none() => {
                push_parse_output_by_index(
                    &mut field_values,
                    &mut complements,
                    capture_names,
                    None,
                );
            }
            _ => {
                polars_bail!(
                    InvalidOperation:
                    "tokenized struct row {} has inconsistent nullability across fields",
                    index
                )
            }
        }
    }

    Ok(ChunkExtractOutput {
        field_values,
        complements,
        compact_profile,
    })
}

fn parse_from_tokenized_parts<T: AsRef<str>, C: AsRef<str>>(
    extractor: &Extractor,
    raw_value: impl AsRef<str>,
    tokens: &[T],
    classes: &[C],
    compiled_pattern: &CompiledPattern,
    mode: MatchMode,
) -> PolarsResult<ParseOutput> {
    // Uses the precompiled pattern so the per-row hot loop does not take the
    // extractor's `compiled_pattern_cache` lock (the pattern is identical for
    // every row of a call). See `compile_extract_pattern`. The `extractor` may be
    // the shared one (serial) or a per-chunk one (parallel) to avoid cache-lock
    // contention across rayon workers.
    extractor
        .parse_compiled_tokens_with_views(
            raw_value.as_ref(),
            tokens,
            classes,
            compiled_pattern,
            mode,
        )
        .map_err(|error| polars_err!(ComputeError: "failed to extract TEL pattern: {}", error))
}

/// Build a fresh `Extractor` that shares the model's token definitions/classes
/// but has its own (empty) plan/regex caches. Used per parallel chunk so rayon
/// workers never contend on the shared extractor's `Mutex`-guarded caches.
fn fresh_chunk_extractor(context: &ModelContext) -> Extractor {
    Extractor::new(
        context.model.token_definitions().clone(),
        context.model.token_class_list().clone(),
    )
    .with_word_definition(context.word_definition.clone())
}

/// Compile the TEL pattern once per extract call (hoisted out of the row loop).
///
/// The model's word definition is baked into the compiled pattern (and carried
/// on it) so neither compilation nor the downstream object-plan builder reads
/// process-global state. This is what keeps concurrent extraction of different
/// models deterministic.
fn compile_extract_pattern(
    pattern: &str,
    word_definition: &WordDefinition,
) -> PolarsResult<CompiledPattern> {
    CompiledPattern::compile_with_word_definition(pattern, word_definition).map_err(
        |error| polars_err!(ComputeError: "failed to compile TEL pattern '{}': {}", pattern, error),
    )
}

/// Borrow the row's string list values as `&str` views into `series` instead of
/// allocating an owned `String` per token. The extraction/parse path is generic
/// over `AsRef<str>`, so views work unchanged -- this removes ~2 owned-String
/// allocations per token per row (the dominant per-row allocation in the hot
/// extraction loop). Requires a `String`-typed list (callers cast categorical
/// token/class columns to `List(String)` upstream).
fn list_series_to_str_views(series: &Series) -> PolarsResult<Vec<&str>> {
    series
        .str()?
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| polars_err!(InvalidOperation: "list values may not contain nulls"))
        })
        .collect()
}

#[cfg(test)]
fn list_series_to_strings(series: &Series) -> PolarsResult<Vec<String>> {
    match series.dtype() {
        DataType::String => series
            .str()?
            .into_iter()
            .map(|value| {
                value.map(ToString::to_string).ok_or_else(
                    || polars_err!(InvalidOperation: "list values may not contain nulls"),
                )
            })
            .collect(),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let casted = series.cast(&DataType::String)?;
            casted
                .str()?
                .into_iter()
                .map(|value| {
                    value.map(ToString::to_string).ok_or_else(
                        || polars_err!(InvalidOperation: "list values may not contain nulls"),
                    )
                })
                .collect()
        }
        dtype => polars_bail!(
            InvalidOperation:
            "expected String, Categorical, or Enum list values, got {:?}",
            dtype
        ),
    }
}

fn list_series_to_u8(series: &Series) -> PolarsResult<Vec<u8>> {
    series
        .u8()?
        .into_iter()
        .map(|value| {
            value.ok_or_else(|| polars_err!(InvalidOperation: "list values may not contain nulls"))
        })
        .collect()
}

fn fill_series_u8(series: &Series, buffer: &mut Vec<u8>) -> PolarsResult<()> {
    buffer.clear();
    buffer.extend(
        series
            .u8()?
            .into_iter()
            .map(|value| {
                value.ok_or_else(
                    || polars_err!(InvalidOperation: "list values may not contain nulls"),
                )
            })
            .collect::<PolarsResult<Vec<_>>>()?,
    );
    Ok(())
}

fn join_string_values<T: AsRef<str>>(values: &[T]) -> String {
    let total_len = values.iter().map(|value| value.as_ref().len()).sum();
    let mut out = String::with_capacity(total_len);
    for value in values {
        out.push_str(value.as_ref());
    }
    out
}

fn profile_enabled() -> bool {
    std::env::var("TOKMAT_PROFILE").is_ok_and(|value| value != "0" && !value.is_empty())
}

/// Minimum row count before splitting work across rayon threads. Below this the
/// thread-coordination overhead outweighs the gain, so we stay single-threaded.
const PARALLEL_ROW_THRESHOLD: usize = 100_000;

/// True when we are already executing on a rayon worker thread — i.e. the caller
/// (e.g. the Polars engine, which parallelizes elementwise expressions across
/// chunks on its own pool) is already parallelizing. In that case the plugin must
/// NOT spawn its own rayon work, or it oversubscribes / nests pools. The internal
/// parallelism is therefore an eager, top-level-only fallback (useful for the
/// direct Rust API on a single-chunk column); when driven through Polars,
/// `is_elementwise` lets the engine do the parallelism instead.
fn running_inside_rayon_pool() -> bool {
    rayon::current_thread_index().is_some()
}

/// Tokenization is embarrassingly parallel and shares no mutable state, so it is
/// parallel by default for large inputs *when called at top level*.
/// `TOKMAT_DISABLE_RAYON` opts out.
fn should_parallelize_tokenize(row_count: usize) -> bool {
    std::env::var_os("TOKMAT_DISABLE_RAYON").is_none()
        && !running_inside_rayon_pool()
        && rayon::current_num_threads() > 1
        && row_count >= PARALLEL_ROW_THRESHOLD
}

/// Extraction is **opt-in** parallel (`TOKMAT_ENABLE_RAYON`). Unlike tokenize,
/// the per-row extract path hammers the `Extractor`'s `Mutex`-guarded plan/regex
/// caches; on high-cardinality (cache-missing) inputs the lock contention makes
/// parallel extraction much slower than serial. It only pays off for
/// low-cardinality workloads whose plans stay cache-resident, so callers must
/// enable it explicitly.
fn should_parallelize_extract(row_count: usize) -> bool {
    std::env::var_os("TOKMAT_ENABLE_RAYON").is_some()
        && std::env::var_os("TOKMAT_DISABLE_RAYON").is_none()
        && !running_inside_rayon_pool()
        && rayon::current_num_threads() > 1
        && row_count >= PARALLEL_ROW_THRESHOLD
}

fn parallel_chunk_size(row_count: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (row_count / (threads * 4)).max(50_000)
}

fn elapsed_since(start: Option<Instant>) -> Duration {
    start.map_or(Duration::ZERO, |start| start.elapsed())
}

fn build_tokenized_struct_series(
    input: &Series,
    columns: TokenizedColumns,
    context: &ModelContext,
    layout: TokenizeLayout,
) -> PolarsResult<Series> {
    let row_count = columns.token_values.len();
    let mut fields = Vec::new();

    if layout.include_raw_value {
        // `raw_value` mirrors the input column verbatim; reuse its Arrow buffers
        // (Arc clone + rename) instead of rebuilding the column.
        let mut raw = input.clone();
        raw.rename("raw_value".into());
        fields.push(raw);
    }

    fields.push(build_output_string_list_series(
        "tokens",
        columns.token_values,
        layout.token_output,
        None,
    )?);

    if let Some(type_values) = columns.type_values {
        fields.push(build_output_string_list_series(
            "types",
            type_values,
            layout.type_output,
            Some(&context.type_enum_values),
        )?);
    }
    if let Some(class_values) = columns.class_values {
        fields.push(build_output_string_list_series(
            "classes",
            class_values,
            layout.class_output,
            Some(&context.class_enum_values),
        )?);
    }
    if let Some(type_id_values) = columns.type_id_values {
        fields.push(build_u8_list_series("type_ids", type_id_values));
    }
    if let Some(class_id_values) = columns.class_id_values {
        fields.push(build_u8_list_series("class_ids", class_id_values));
    }

    Ok(StructChunked::from_series(input.name().clone(), row_count, fields.iter())?.into_series())
}

fn build_extract_struct_series(
    name: PlSmallStr,
    field_columns: Vec<(String, Vec<Option<String>>)>,
    complements: &[Option<String>],
) -> PolarsResult<Series> {
    let row_count = complements.len();

    let mut field_series = field_columns
        .into_iter()
        .map(|(field_name, values)| {
            StringChunked::from_iter_options(
                field_name.into(),
                values.iter().map(|value| value.as_deref()),
            )
            .into_series()
        })
        .collect::<Vec<_>>();

    field_series.push(
        StringChunked::from_iter_options(
            "complement".into(),
            complements.iter().map(|value| value.as_deref()),
        )
        .into_series(),
    );

    Ok(StructChunked::from_series(name, row_count, field_series.iter())?.into_series())
}

/// Streams serial-path extract results straight into Arrow string builders, one
/// per capture field plus the complement. This avoids materializing the whole
/// output column set as an intermediate `Vec<Option<String>>` (which doubled
/// peak memory and added a copy): each captured value is *moved* out of the
/// `ParseOutput` map and written once into the Arrow buffer.
struct ExtractColumnBuilders {
    fields: Vec<StringChunkedBuilder>,
    complement: StringChunkedBuilder,
}

impl ExtractColumnBuilders {
    fn new(capture_names: &[String], row_count: usize) -> Self {
        Self {
            fields: capture_names
                .iter()
                .map(|name| StringChunkedBuilder::new(name.clone().into(), row_count))
                .collect(),
            complement: StringChunkedBuilder::new("complement".into(), row_count),
        }
    }

    fn push(&mut self, capture_names: &[String], output: Option<ParseOutput>) {
        if let Some(mut output) = output {
            for (builder, name) in self.fields.iter_mut().zip(capture_names) {
                match output.fields.remove(name) {
                    Some(value) => builder.append_value(value),
                    None => builder.append_null(),
                }
            }
            self.complement.append_value(output.complement);
        } else {
            for builder in &mut self.fields {
                builder.append_null();
            }
            self.complement.append_null();
        }
    }

    fn finish(self, name: PlSmallStr, row_count: usize) -> PolarsResult<Series> {
        let mut field_series = self
            .fields
            .into_iter()
            .map(|builder| builder.finish().into_series())
            .collect::<Vec<_>>();
        field_series.push(self.complement.finish().into_series());
        Ok(StructChunked::from_series(name, row_count, field_series.iter())?.into_series())
    }
}

fn push_parse_output_by_index(
    field_values: &mut [Vec<Option<String>>],
    complements: &mut Vec<Option<String>>,
    capture_names: &[String],
    output: Option<ParseOutput>,
) {
    if let Some(output) = output {
        for (index, name) in capture_names.iter().enumerate() {
            field_values[index].push(output.fields.get(name).cloned());
        }
        complements.push(Some(output.complement));
    } else {
        for values in field_values.iter_mut() {
            values.push(None);
        }
        complements.push(None);
    }
}

fn build_string_list_series(name: &str, rows: Vec<Option<Vec<String>>>) -> Series {
    let row_count = rows.len();
    let values_capacity = rows
        .iter()
        .flatten()
        .map(|values| values.iter().map(String::len).sum::<usize>())
        .sum();
    let mut builder = ListStringChunkedBuilder::new(name.into(), row_count, values_capacity);
    for row in rows {
        match row {
            Some(values) => builder.append_values_iter(values.iter().map(String::as_str)),
            None => builder.append_null(),
        }
    }
    builder.finish().into_series()
}

#[allow(unsafe_code)]
fn build_enum_list_series(
    name: &str,
    rows: Vec<Option<Vec<String>>>,
    enum_values: &[String],
) -> PolarsResult<Series> {
    let enum_dtype = enum_dtype(enum_values);
    let rows = rows
        .into_iter()
        .map(|row| {
            row.map(|values| Series::new(PlSmallStr::EMPTY, values).cast(&enum_dtype))
                .transpose()
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let base = rows.into_iter().collect::<ListChunked>().into_series();
    let list_dtype = DataType::List(Box::new(enum_dtype));
    // SAFETY:
    // `base` already contains valid list chunks; we are only restoring the logical
    // list child dtype metadata so the outer struct field matches the produced value.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(name.into(), base.chunks().clone(), &list_dtype)
    })
}

fn build_u8_list_series(name: &str, rows: Vec<Option<Vec<u8>>>) -> Series {
    let row_count = rows.len();
    let values_capacity = rows.iter().flatten().map(Vec::len).sum();
    let mut builder = ListPrimitiveChunkedBuilder::<UInt8Type>::new(
        name.into(),
        row_count,
        values_capacity,
        DataType::UInt8,
    );
    for row in rows {
        match row {
            Some(values) => builder.append_slice(&values),
            None => builder.append_null(),
        }
    }
    builder.finish().into_series()
}

/// ASCII whitespace per Unicode `White_Space` (matches `char::is_whitespace`
/// for ASCII, which includes vertical tab `0x0B` unlike `u8::is_ascii_whitespace`).
const fn ascii_is_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

fn classify_token_ref<'a>(
    token: &'a str,
    model: &'a TokenModel,
    features: ModelFeatures,
) -> Cow<'a, str> {
    if token.is_empty() {
        return Cow::Borrowed(token);
    }

    // Fast-path classification: every shortcut below requires an ASCII token, so
    // compute all character-class predicates in a single byte pass instead of
    // re-walking `token.chars()` (and re-checking `is_ascii`) up to ten times,
    // and validate the POSTALCODE shape without allocating a `String`/`Vec`.
    if token.is_ascii() {
        let mut all_digit = true;
        let mut all_alpha = true;
        let mut has_digit = false;
        let mut has_alpha = false;
        let mut all_digit_or_dash = true;
        let mut all_alpha_dash_apos = true;
        let mut all_alnum = true;
        let mut all_alnum_dash_apos = true;

        // Compacted (whitespace/'-'/'_' stripped) characters for the A1A1A1
        // POSTALCODE check; capped at 7 so we can cheaply reject anything > 6.
        let mut postal = [0_u8; 6];
        let mut postal_len = 0_usize;
        let mut postal_overflow = false;

        for &byte in token.as_bytes() {
            let is_digit = byte.is_ascii_digit();
            let is_alpha = byte.is_ascii_alphabetic();
            let is_dash = byte == b'-';
            let is_apos = byte == b'\'';
            let is_alnum = is_digit | is_alpha;

            has_digit |= is_digit;
            has_alpha |= is_alpha;
            all_digit &= is_digit;
            all_alpha &= is_alpha;
            all_digit_or_dash &= is_digit | is_dash;
            all_alpha_dash_apos &= is_alpha | is_dash | is_apos;
            all_alnum &= is_alnum;
            all_alnum_dash_apos &= is_alnum | is_dash | is_apos;

            if features.has_postalcode
                && !postal_overflow
                && !is_dash
                && byte != b'_'
                && !ascii_is_whitespace(byte)
            {
                if postal_len < postal.len() {
                    postal[postal_len] = byte;
                    postal_len += 1;
                } else {
                    postal_overflow = true;
                }
            }
        }

        if features.has_postalcode
            && !postal_overflow
            && postal_len == 6
            && postal[0].is_ascii_alphabetic()
            && postal[1].is_ascii_digit()
            && postal[2].is_ascii_alphabetic()
            && postal[3].is_ascii_digit()
            && postal[4].is_ascii_alphabetic()
            && postal[5].is_ascii_digit()
        {
            return Cow::Borrowed("POSTALCODE");
        }
        if all_digit && features.has_num {
            return Cow::Borrowed("NUM");
        }
        if all_alpha && features.has_alpha {
            return Cow::Borrowed("ALPHA");
        }
        if all_digit_or_dash && has_digit && features.has_num_extended {
            return Cow::Borrowed("NUM_EXTENDED");
        }
        if all_alpha_dash_apos && has_alpha && features.has_alpha_extended {
            return Cow::Borrowed("ALPHA_EXTENDED");
        }
        if all_alnum_dash_apos && has_alpha && has_digit {
            if all_alnum && features.has_alpha_num {
                return Cow::Borrowed("ALPHA_NUM");
            }
            if features.has_alpha_num_extended {
                return Cow::Borrowed("ALPHA_NUM_EXTENDED");
            }
        }
    }

    model
        .compiled_patterns()
        .iter()
        .find_map(|(name, regex)| {
            regex
                .is_match(token.as_bytes())
                .ok()
                .and_then(|matched| matched.then_some(Cow::Borrowed(name.as_str())))
        })
        .unwrap_or(Cow::Borrowed(token))
}

fn tokenize_row(raw_value: &str, context: &ModelContext, layout: TokenizeLayout) -> TokenizedRow {
    let tokens = split_input_tokens_with(raw_value, context.model.word_boundary());
    let mut type_values = layout
        .needs_type_values()
        .then(|| Vec::with_capacity(tokens.len()));
    let mut class_values = layout
        .include_classes
        .then(|| Vec::with_capacity(tokens.len()));
    let mut type_ids = layout
        .include_type_ids
        .then(|| Vec::with_capacity(tokens.len()));
    let mut class_ids = layout
        .include_class_ids
        .then(|| Vec::with_capacity(tokens.len()));

    for token in &tokens {
        let token_type = classify_token_ref(token, &context.model, context.features).into_owned();
        if let Some(values) = type_values.as_mut() {
            values.push(token_type.clone());
        }
        if let Some(values) = type_ids.as_mut() {
            values.push(context.type_codec.encode_known_or_raw(&token_type));
        }

        let class_value = if token.chars().all(char::is_whitespace) {
            Cow::Borrowed(token.as_str())
        } else if let Some(value) = context.model.token_class_lookup().get(token) {
            Cow::Borrowed(value.as_str())
        } else {
            Cow::Borrowed(token_type.as_str())
        };

        if let Some(values) = class_values.as_mut() {
            values.push(class_value.to_string());
        }
        if let Some(values) = class_ids.as_mut() {
            values.push(
                context
                    .class_codec
                    .encode_known_or_raw(class_value.as_ref()),
            );
        }
    }

    TokenizedRow {
        tokens,
        types: type_values,
        classes: class_values,
        type_ids,
        class_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_model_path() -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/model_1")
            .to_string_lossy()
            .into_owned()
    }

    fn legacy_tokenize_kwargs() -> TokenizeKwargs {
        TokenizeKwargs {
            model_path: fixture_model_path(),
            include_raw_value: true,
            include_types: true,
            include_classes: true,
            include_type_ids: false,
            include_class_ids: false,
            token_output: StringListOutput::String,
            type_output: StringListOutput::String,
            class_output: StringListOutput::String,
        }
    }

    fn string_field_at(series: &Series, name: &str, row: usize) -> Option<String> {
        let struct_chunked = series.struct_().expect("output should be a struct");
        let fields = struct_chunked.fields_as_series();
        fields
            .iter()
            .find(|field| field.name().as_str() == name)
            .unwrap_or_else(|| panic!("{name} field should exist"))
            .str()
            .unwrap_or_else(|_| panic!("{name} field should be string"))
            .get(row)
            .map(ToString::to_string)
    }

    fn struct_field_names(series: &Series) -> Vec<String> {
        series
            .struct_()
            .expect("output should be a struct")
            .fields_as_series()
            .iter()
            .map(|field| field.name().to_string())
            .collect()
    }

    #[test]
    fn tokenize_helper_returns_struct_with_expected_fields() {
        let input = Series::new("address".into(), &[Some("123 MAIN ST"), None]);
        let output = tokenize_expr_impl(&[input], &legacy_tokenize_kwargs())
            .expect("tokenize should succeed");

        let struct_chunked = output.struct_().expect("tokenize output should be struct");
        let fields = struct_chunked.fields_as_series();
        let field_names = fields
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(field_names, vec!["raw_value", "tokens", "types", "classes"]);

        let token_field = fields
            .iter()
            .find(|field| field.name().as_str() == "tokens")
            .expect("tokens field should exist");
        let first_tokens = token_field
            .list()
            .expect("tokens should be a list")
            .into_iter()
            .next()
            .expect("first row should exist")
            .expect("first row should be non-null");
        let token_values =
            list_series_to_strings(&first_tokens).expect("list conversion should work");
        assert!(token_values.contains(&"123".to_string()));
        assert!(token_values.contains(&"MAIN".to_string()));
    }

    #[test]
    fn tokenize_helper_can_emit_compact_class_ids() {
        let input = Series::new("address".into(), ["123 MAIN ST"]);
        let output = tokenize_expr_impl(
            &[input],
            &TokenizeKwargs {
                model_path: fixture_model_path(),
                include_raw_value: false,
                include_types: false,
                include_classes: false,
                include_type_ids: false,
                include_class_ids: true,
                token_output: StringListOutput::String,
                type_output: StringListOutput::String,
                class_output: StringListOutput::String,
            },
        )
        .expect("compact tokenize should succeed");

        let struct_chunked = output.struct_().expect("tokenize output should be struct");
        let fields = struct_chunked.fields_as_series();
        let field_names = fields
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>();
        assert_eq!(field_names, vec!["tokens", "class_ids"]);
    }

    #[test]
    fn tokenize_helper_can_emit_categorical_lists() {
        let input = Series::new("address".into(), ["123 MAIN ST"]);
        let output = tokenize_expr_impl(
            &[input],
            &TokenizeKwargs {
                model_path: fixture_model_path(),
                include_raw_value: false,
                include_types: true,
                include_classes: true,
                include_type_ids: false,
                include_class_ids: false,
                token_output: StringListOutput::Categorical,
                type_output: StringListOutput::Categorical,
                class_output: StringListOutput::Categorical,
            },
        )
        .expect("categorical tokenize should succeed");

        let struct_chunked = output.struct_().expect("tokenize output should be struct");
        let fields = struct_chunked.fields_as_series();
        assert_eq!(
            fields[0].dtype(),
            &DataType::List(Box::new(DataType::Categorical(
                None,
                CategoricalOrdering::default(),
            )))
        );
        assert_eq!(
            fields[1].dtype(),
            &DataType::List(Box::new(DataType::Categorical(
                None,
                CategoricalOrdering::default(),
            )))
        );
        assert_eq!(
            fields[2].dtype(),
            &DataType::List(Box::new(DataType::Categorical(
                None,
                CategoricalOrdering::default(),
            )))
        );
    }

    #[test]
    fn extract_helper_accepts_raw_string_input() {
        let input = Series::new("address".into(), ["123 MAIN ST"]);
        let output = extract_expr_impl(
            &[input],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::default(),
            },
        )
        .expect("extract should succeed");

        let struct_chunked = output.struct_().expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let civic = fields
            .iter()
            .find(|field| field.name().as_str() == "CIVIC")
            .expect("CIVIC field should exist")
            .str()
            .expect("CIVIC field should be string")
            .get(0);
        let street = fields
            .iter()
            .find(|field| field.name().as_str() == "STREET")
            .expect("STREET field should exist")
            .str()
            .expect("STREET field should be string")
            .get(0);
        let street_type = fields
            .iter()
            .find(|field| field.name().as_str() == "TYPE")
            .expect("TYPE field should exist")
            .str()
            .expect("TYPE field should be string")
            .get(0);

        assert_eq!(civic, Some("123"));
        assert_eq!(street, Some("MAIN"));
        assert_eq!(street_type, Some("ST"));
    }

    #[test]
    fn extract_helper_accepts_tokenized_struct_input() {
        let tokenized = tokenize_expr_impl(
            &[Series::new("address".into(), ["123 MAIN ST"])],
            &legacy_tokenize_kwargs(),
        )
        .expect("tokenize should succeed");

        let output = extract_expr_impl(
            &[tokenized],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::default(),
            },
        )
        .expect("extract should succeed");

        let struct_chunked = output.struct_().expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let complement = fields
            .iter()
            .find(|field| field.name().as_str() == "complement")
            .expect("complement field should exist")
            .str()
            .expect("complement should be string")
            .get(0);

        assert_eq!(complement, Some(""));
    }

    #[test]
    fn extract_helper_accepts_tokenized_struct_without_raw_value_or_types() {
        let tokens = build_string_list_series(
            "tokens",
            [Some(vec![
                "123".to_string(),
                " ".to_string(),
                "MAIN".to_string(),
                " ".to_string(),
                "ST".to_string(),
            ])]
            .to_vec(),
        );
        let classes = build_string_list_series(
            "classes",
            [Some(vec![
                "NUM".to_string(),
                " ".to_string(),
                "ALPHA".to_string(),
                " ".to_string(),
                "STREETTYPE".to_string(),
            ])]
            .to_vec(),
        );
        let tokenized = StructChunked::from_series("address".into(), 1, [tokens, classes].iter())
            .expect("struct should build")
            .into_series();

        let output = extract_expr_impl(
            &[tokenized],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::default(),
            },
        )
        .expect("extract should succeed");

        let struct_chunked = output.struct_().expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let civic = fields
            .iter()
            .find(|field| field.name().as_str() == "CIVIC")
            .expect("CIVIC field should exist")
            .str()
            .expect("CIVIC field should be string")
            .get(0);

        assert_eq!(civic, Some("123"));
    }

    #[test]
    fn extract_helper_accepts_tokenized_struct_with_class_ids() {
        let tokenized = tokenize_expr_impl(
            &[Series::new("address".into(), ["123 MAIN ST"])],
            &TokenizeKwargs {
                model_path: fixture_model_path(),
                include_raw_value: false,
                include_types: false,
                include_classes: false,
                include_type_ids: false,
                include_class_ids: true,
                token_output: StringListOutput::String,
                type_output: StringListOutput::String,
                class_output: StringListOutput::String,
            },
        )
        .expect("compact tokenize should succeed");

        let output = extract_expr_impl(
            &[tokenized],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::default(),
            },
        )
        .expect("extract should succeed");

        let struct_chunked = output.struct_().expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let civic = fields
            .iter()
            .find(|field| field.name().as_str() == "CIVIC")
            .expect("CIVIC field should exist")
            .str()
            .expect("CIVIC field should be string")
            .get(0);

        assert_eq!(civic, Some("123"));
    }

    #[test]
    fn extract_helper_respects_any_mode_for_raw_string_input() {
        let input = Series::new("address".into(), ["ATTN 123 MAIN ST"]);
        let output = extract_expr_impl(
            &[input],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::Any,
            },
        )
        .expect("extract should succeed");

        let struct_chunked = output.struct_().expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let civic = fields
            .iter()
            .find(|field| field.name().as_str() == "CIVIC")
            .expect("CIVIC field should exist")
            .str()
            .expect("CIVIC field should be string")
            .get(0);
        let complement = fields
            .iter()
            .find(|field| field.name().as_str() == "complement")
            .expect("complement field should exist")
            .str()
            .expect("complement should be string")
            .get(0);

        assert_eq!(civic, Some("123"));
        assert_eq!(complement, Some("ATTN "));
    }

    #[test]
    fn extract_helper_drops_capture_like_vanishing_groups() {
        let input = Series::new("address".into(), ["ATTN 123 MAIN ST"]);
        let output = extract_expr_impl(
            &[input],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<!DROP@!> <<CIVIC::NUM>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::Whole,
            },
        )
        .expect("capture-like vanishing extract should succeed");

        assert_eq!(string_field_at(&output, "CIVIC", 0).as_deref(), Some("123"));
        assert_eq!(
            string_field_at(&output, "STREET", 0).as_deref(),
            Some("MAIN")
        );
        assert_eq!(string_field_at(&output, "TYPE", 0).as_deref(), Some("ST"));
        assert_eq!(
            string_field_at(&output, "complement", 0).as_deref(),
            Some("ATTN ")
        );
        assert!(
            !struct_field_names(&output)
                .iter()
                .any(|name| name == "DROP"),
            "vanishing group labels must not become output fields"
        );
    }

    #[test]
    fn extract_helper_applies_optional_literal_vanishing_to_tokenized_structs() {
        let tokenized = tokenize_expr_impl(
            &[Series::new(
                "address".into(),
                ["#123 MAIN ST", "123 MAIN ST"],
            )],
            &legacy_tokenize_kwargs(),
        )
        .expect("tokenize should succeed");

        let output = extract_expr_impl(
            &[tokenized],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<!{{#}}?!> <<CIVIC::NUM>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::Whole,
            },
        )
        .expect("optional literal vanishing extract should succeed");

        assert_eq!(string_field_at(&output, "CIVIC", 0).as_deref(), Some("123"));
        assert_eq!(
            string_field_at(&output, "STREET", 0).as_deref(),
            Some("MAIN")
        );
        assert_eq!(string_field_at(&output, "TYPE", 0).as_deref(), Some("ST"));
        assert_eq!(
            string_field_at(&output, "complement", 0).as_deref(),
            Some("#")
        );
        assert_eq!(string_field_at(&output, "CIVIC", 1).as_deref(), Some("123"));
        assert_eq!(
            string_field_at(&output, "STREET", 1).as_deref(),
            Some("MAIN")
        );
        assert_eq!(string_field_at(&output, "TYPE", 1).as_deref(), Some("ST"));
        assert_eq!(
            string_field_at(&output, "complement", 1).as_deref(),
            Some("")
        );
    }

    #[test]
    fn extract_helper_applies_anonymous_optional_vanishing_groups() {
        let input = Series::new("address".into(), ["ATTN 123 MAIN ST", "123 MAIN ST"]);
        let output = extract_expr_impl(
            &[input],
            ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<!?!> <<CIVIC::NUM>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
                mode: MatchModeKwarg::Whole,
            },
        )
        .expect("anonymous optional vanishing extract should succeed");

        assert_eq!(string_field_at(&output, "CIVIC", 0).as_deref(), Some("123"));
        assert_eq!(
            string_field_at(&output, "STREET", 0).as_deref(),
            Some("MAIN")
        );
        assert_eq!(string_field_at(&output, "TYPE", 0).as_deref(), Some("ST"));
        assert_eq!(string_field_at(&output, "CIVIC", 1).as_deref(), Some("123"));
        assert_eq!(
            string_field_at(&output, "STREET", 1).as_deref(),
            Some("MAIN")
        );
        assert_eq!(string_field_at(&output, "TYPE", 1).as_deref(), Some("ST"));
    }

    #[test]
    fn per_model_word_definition_changes_tokenization() {
        let wd_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/model_worddef")
            .to_string_lossy()
            .into_owned();
        // Load both models. The custom model's `WORDDEFINITION.param` adds `.`
        // as a word char; the default does not. Because tokenization now uses the
        // per-model boundary (not the process-global), results are deterministic
        // regardless of which model was loaded last.
        let default_plugin =
            TokmatPolars::from_model_path(fixture_model_path()).expect("model_1 loads");
        let wd_plugin = TokmatPolars::from_model_path(&wd_path).expect("model_worddef loads");

        let input = Series::new("address".into(), ["192.168 MAIN"]);
        let first_tokens = |plugin: &TokmatPolars| -> Vec<String> {
            let out = plugin.tokenize_series(&input).expect("tokenize");
            let st = out.struct_().expect("struct output");
            let fields = st.fields_as_series();
            let tokens = fields
                .iter()
                .find(|f| f.name().as_str() == "tokens")
                .expect("tokens field");
            let first = tokens
                .list()
                .expect("list")
                .into_iter()
                .next()
                .flatten()
                .expect("first row");
            list_series_to_strings(&first).expect("token strings")
        };

        let wd_tokens = first_tokens(&wd_plugin);
        let default_tokens = first_tokens(&default_plugin);

        // Custom def: `.` is a word char, so "192.168" stays one token.
        assert!(
            wd_tokens.iter().any(|t| t == "192.168"),
            "custom-word-def tokens: {wd_tokens:?}"
        );
        // Default def: `.` is a boundary, so "192.168" splits.
        assert!(
            !default_tokens.iter().any(|t| t == "192.168")
                && default_tokens.iter().any(|t| t == "192"),
            "default tokens: {default_tokens:?}"
        );
    }

    fn worddef_model_path() -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/model_worddef")
            .to_string_lossy()
            .into_owned()
    }

    // Regression guard for the process-global word-definition race
    // (`wanparser-streaming-engine-nondeterminism`): extraction must compile
    // with each model's own word definition, threaded onto the compiled pattern
    // and the extractor, never via process-global state. A revert that drops
    // `with_word_definition` or routes compilation back through the global-reading
    // `CompiledPattern::compile` fails these assertions.
    #[test]
    fn extraction_wires_per_model_word_definition_onto_context_and_pattern() {
        let default_ctx = get_or_load_context(&fixture_model_path()).expect("model_1 loads");
        let wd_ctx = get_or_load_context(&worddef_model_path()).expect("model_worddef loads");

        // The context's model definition and the per-model extractor agree.
        assert_eq!(default_ctx.word_definition.chars(), r"\w\-'");
        assert_eq!(default_ctx.extractor.word_definition().chars(), r"\w\-'");
        assert_eq!(wd_ctx.word_definition.chars(), r"\w\-'.");
        assert_eq!(wd_ctx.extractor.word_definition().chars(), r"\w\-'.");

        // Compiling an extract pattern bakes the model's definition onto the
        // pattern (so the downstream object-plan builder reuses the exact same
        // word-boundary class).
        let pattern = "<<NAME>>";
        let default_compiled =
            compile_extract_pattern(pattern, &default_ctx.word_definition).expect("compiles");
        let wd_compiled =
            compile_extract_pattern(pattern, &wd_ctx.word_definition).expect("compiles");

        assert_eq!(default_compiled.word_definition().chars(), r"\w\-'");
        assert_eq!(wd_compiled.word_definition().chars(), r"\w\-'.");
        assert!(
            wd_compiled
                .class_pattern(tokmat::tel::MatchMode::Whole)
                .contains(r"\w\-'."),
            "model word definition must be baked into the class comparator: {}",
            wd_compiled.class_pattern(tokmat::tel::MatchMode::Whole)
        );
        assert!(
            !default_compiled
                .class_pattern(tokmat::tel::MatchMode::Whole)
                .contains(r"\w\-'."),
        );
    }

    // End-to-end guard for the write-race: extracting with one model must not
    // perturb extraction with another model, regardless of order. Before the fix
    // each extract wrote the model's definition to a process-global, so
    // interleaving models with different definitions produced order-dependent
    // results.
    #[test]
    fn extraction_is_order_independent_across_models() {
        let default_plugin =
            TokmatPolars::from_model_path(fixture_model_path()).expect("model_1 loads");
        let wd_plugin =
            TokmatPolars::from_model_path(worddef_model_path()).expect("model_worddef loads");

        let input = Series::new("address".into(), ["192.168 MAIN"]);
        let pattern = "<<FIRST%>>";
        let extract = |plugin: &TokmatPolars| -> Series {
            plugin
                .extract_series_with_mode(&input, pattern, MatchMode::Start)
                .expect("extract")
        };

        // Baselines, each model alone.
        let wd_alone = extract(&wd_plugin);
        let default_alone = extract(&default_plugin);

        // Interleave: a default extraction between two word-def extractions must
        // not change the word-def result.
        let _ = extract(&default_plugin);
        let wd_after_default = extract(&wd_plugin);
        let _ = extract(&wd_plugin);
        let default_after_wd = extract(&default_plugin);

        assert!(
            wd_alone.equals(&wd_after_default),
            "word-def extraction must be independent of interleaved other-model extraction",
        );
        assert!(
            default_alone.equals(&default_after_wd),
            "default extraction must be independent of interleaved other-model extraction",
        );
        // The two models genuinely consult different word definitions, so the
        // order-independence above is a meaningful property, not a vacuous one.
        assert!(
            !wd_alone.equals(&default_alone),
            "models with different word definitions must extract differently",
        );
    }

    #[test]
    fn parallel_tokenize_matches_serial_across_chunks() {
        let context = get_or_load_context(&fixture_model_path()).expect("model should load");
        let layout = TokenizeLayout {
            include_raw_value: true,
            include_types: true,
            include_classes: true,
            include_type_ids: true,
            include_class_ids: true,
            token_output: StringListOutput::String,
            type_output: StringListOutput::String,
            class_output: StringListOutput::String,
        };
        let input = Series::new(
            "address".into(),
            &[
                Some("123 MAIN ST"),
                None,
                Some("4567 OAK AVE NW"),
                Some("89 KING ST W"),
                Some("K1A 0B1"),
                None,
                Some("12-34 RIVER RD"),
                Some("O'CONNOR BLVD"),
                Some("1 A ST"),
                Some("PO BOX 42"),
            ],
        );

        let serial = tokenize_series_serial(&input, &context, layout).expect("serial tokenize");
        // chunk_size = 3 forces multiple chunks + the null-boundary cases.
        let chunked =
            tokenize_series_chunked(&input, &context, layout, 3).expect("chunked tokenize");

        assert!(
            serial.equals_missing(&chunked),
            "parallel/chunked tokenize output must equal serial output"
        );
    }

    #[test]
    fn classify_token_ref_covers_fast_path_branches() {
        let context = get_or_load_context(&fixture_model_path()).expect("model should load");
        let classify =
            |token: &str| classify_token_ref(token, &context.model, context.features).into_owned();

        // POSTALCODE shape (A1A1A1), including dash/underscore/space stripping.
        assert_eq!(classify("K1A0B1"), "POSTALCODE");
        assert_eq!(classify("K1A-0B1"), "POSTALCODE");
        assert_eq!(classify("K1A_0B1"), "POSTALCODE");
        // Not a postal code: wrong length / shape.
        assert_ne!(classify("K1A0B12"), "POSTALCODE");
        assert_ne!(classify("KKA0B1"), "POSTALCODE");

        // Pure digit / pure alpha.
        assert_eq!(classify("123"), "NUM");
        assert_eq!(classify("MAIN"), "ALPHA");

        // Extended classes.
        assert_eq!(classify("12-34"), "NUM_EXTENDED");
        assert_eq!(classify("O'BRIEN"), "ALPHA_EXTENDED");
        assert_eq!(classify("ST-PAUL"), "ALPHA_EXTENDED");

        // Mixed alphanumerics.
        assert_eq!(classify("12A"), "ALPHA_NUM");
        assert_eq!(classify("12A-B"), "ALPHA_NUM_EXTENDED");

        // Non-ASCII falls through to the regex path (never a fast-path class).
        let allee = classify("ALLÉE");
        assert!(allee != "ALPHA" && allee != "NUM");
    }

    #[test]
    fn rust_api_tokenizes_and_extracts() {
        let plugin =
            TokmatPolars::from_model_path(fixture_model_path()).expect("model should load");
        let input = Series::new("address".into(), ["123 MAIN ST"]);

        let tokenized = plugin
            .tokenize_series(&input)
            .expect("tokenize via rust api");
        let extracted = plugin
            .extract_series(&tokenized, "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>")
            .expect("extract via rust api");

        let struct_chunked = extracted
            .struct_()
            .expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let civic = fields
            .iter()
            .find(|field| field.name().as_str() == "CIVIC")
            .expect("CIVIC field should exist")
            .str()
            .expect("CIVIC field should be string")
            .get(0);

        assert_eq!(civic, Some("123"));
        assert_eq!(
            plugin
                .capture_field_names("<!DROP@!> <<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>")
                .expect("capture names"),
            vec![
                "CIVIC".to_string(),
                "STREET".to_string(),
                "TYPE".to_string()
            ]
        );
    }

    #[test]
    fn rust_api_extracts_with_explicit_match_mode() {
        let plugin =
            TokmatPolars::from_model_path(fixture_model_path()).expect("model should load");
        let input = Series::new("address".into(), ["ATTN 123 MAIN ST"]);

        let extracted = plugin
            .extract_series_with_mode(
                &input,
                "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>",
                MatchMode::Any,
            )
            .expect("extract via rust api");

        let struct_chunked = extracted
            .struct_()
            .expect("extract output should be struct");
        let fields = struct_chunked.fields_as_series();
        let complement = fields
            .iter()
            .find(|field| field.name().as_str() == "complement")
            .expect("complement field should exist")
            .str()
            .expect("complement field should be string")
            .get(0);

        assert_eq!(complement, Some("ATTN "));
    }
}
