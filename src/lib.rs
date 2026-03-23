#![doc = include_str!("../README.md")]

//! Polars integration for tokmat, usable from both Rust and Python.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use tokmat::extractor::{Extractor, MatchMode, ParseOutput};
use tokmat::tel::CompiledPattern;
use tokmat::token_model::TokenModel;
use tokmat::tokenizer::tokenize_with_model;

static CONTEXT_CACHE: LazyLock<Mutex<HashMap<String, Arc<ModelContext>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct ModelContext {
    model: TokenModel,
    extractor: Extractor,
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
        tokenize_series_with_context(input, &self.context)
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
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizeKwargs {
    model_path: String,
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

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func=tokenize_output_type)]
fn tokenize_expr(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    tokenize_expr_impl(inputs, &kwargs)
}

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func_with_kwargs=extract_output_type)]
fn extract_expr(inputs: &[Series], kwargs: ExtractKwargs) -> PolarsResult<Series> {
    extract_expr_impl(inputs, kwargs)
}

#[allow(clippy::unnecessary_wraps)]
#[pymodule]
fn tokmat_polars(_py: Python<'_>, _module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[allow(clippy::unnecessary_wraps)]
fn tokenize_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let output_name = output_field_name(input_fields, "tokenized");
    Ok(Field::new(output_name, DataType::Struct(tokenize_fields())))
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

fn tokenize_expr_impl(inputs: &[Series], kwargs: &TokenizeKwargs) -> PolarsResult<Series> {
    let input = single_input(inputs, "tokenize_expr")?;
    let context = get_or_load_context(&kwargs.model_path)?;
    tokenize_series_with_context(input, &context)
}

fn extract_expr_impl(inputs: &[Series], kwargs: ExtractKwargs) -> PolarsResult<Series> {
    let input = single_input(inputs, "extract_expr")?;
    let context = get_or_load_context(&kwargs.model_path)?;
    extract_series_with_context(input, &context, &kwargs.pattern, kwargs.mode.into())
}

fn tokenize_series_with_context(input: &Series, context: &ModelContext) -> PolarsResult<Series> {
    let row_count = input.len();
    let mut raw_values = Vec::with_capacity(row_count);
    let mut token_values = Vec::with_capacity(row_count);
    let mut type_values = Vec::with_capacity(row_count);
    let mut class_values = Vec::with_capacity(row_count);

    for value in input.str()? {
        if let Some(raw_value) = value {
            let tokenized = tokenize_with_model(raw_value, &context.model);
            raw_values.push(Some(tokenized.raw_value));
            token_values.push(Some(tokenized.tokens));
            type_values.push(Some(tokenized.types));
            class_values.push(Some(tokenized.classes));
        } else {
            raw_values.push(None);
            token_values.push(None);
            type_values.push(None);
            class_values.push(None);
        }
    }

    build_tokenized_struct_series(
        input.name().clone(),
        &raw_values,
        token_values,
        type_values,
        class_values,
    )
}

fn extract_series_with_context(
    input: &Series,
    context: &ModelContext,
    pattern: &str,
    mode: MatchMode,
) -> PolarsResult<Series> {
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
    );

    let context = Arc::new(ModelContext { model, extractor });
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

fn tokenize_fields() -> Vec<Field> {
    vec![
        Field::new("raw_value".into(), DataType::String),
        Field::new("tokens".into(), DataType::List(Box::new(DataType::String))),
        Field::new("types".into(), DataType::List(Box::new(DataType::String))),
        Field::new("classes".into(), DataType::List(Box::new(DataType::String))),
    ]
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
    let mut field_columns = init_extract_columns(capture_names, input.len());
    let mut complements = Vec::with_capacity(input.len());

    for raw_value in input.str()? {
        match raw_value {
            Some(raw_value) => {
                let tokenized = tokenize_with_model(raw_value, &context.model);
                let parsed = parse_from_tokenized_parts(
                    context,
                    raw_value,
                    &tokenized.tokens,
                    &tokenized.classes,
                    pattern,
                    mode,
                )?;
                push_parse_output(&mut field_columns, &mut complements, Some(parsed));
            }
            None => push_parse_output(&mut field_columns, &mut complements, None),
        }
    }

    build_extract_struct_series(input.name().clone(), field_columns, &complements)
}

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
    let classes_field = field_map.get("classes").ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "tokenized struct is missing required 'classes' field"
        )
    })?;

    let raw_values = raw_field
        .map(|field| -> PolarsResult<Vec<Option<String>>> {
            Ok(field
                .str()?
                .into_iter()
                .map(|value| value.map(ToString::to_string))
                .collect::<Vec<_>>())
        })
        .transpose()?;
    let token_lists = tokens_field.list()?.into_iter().collect::<Vec<_>>();
    let class_lists = classes_field.list()?.into_iter().collect::<Vec<_>>();
    let mut field_columns = init_extract_columns(capture_names, input.len());
    let mut complements = Vec::with_capacity(input.len());

    for index in 0..input.len() {
        match (&token_lists[index], &class_lists[index]) {
            (Some(tokens), Some(classes)) => {
                let token_values = list_series_to_strings(tokens)?;
                let class_values = list_series_to_strings(classes)?;
                let raw_value = raw_values
                    .as_ref()
                    .and_then(|values| values[index].clone())
                    .unwrap_or_else(|| token_values.join(""));
                let parsed = parse_from_tokenized_parts(
                    context,
                    &raw_value,
                    &token_values,
                    &class_values,
                    pattern,
                    mode,
                )?;
                push_parse_output(&mut field_columns, &mut complements, Some(parsed));
            }
            (None, None) if raw_values.as_ref().is_none_or(|values| values[index].is_none()) => {
                push_parse_output(&mut field_columns, &mut complements, None);
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

    build_extract_struct_series(input.name().clone(), field_columns, &complements)
}

fn parse_from_tokenized_parts(
    context: &ModelContext,
    raw_value: &str,
    tokens: &[String],
    classes: &[String],
    pattern: &str,
    mode: MatchMode,
) -> PolarsResult<ParseOutput> {
    context
        .extractor
        .parse_tokens(raw_value, tokens, classes, pattern, mode)
        .map_err(|error| {
            polars_err!(
                ComputeError:
                "failed to extract TEL pattern '{}': {}",
                pattern,
                error
            )
        })
}

fn list_series_to_strings(series: &Series) -> PolarsResult<Vec<String>> {
    series
        .str()?
        .into_iter()
        .map(|value| {
            value
                .map(ToString::to_string)
                .ok_or_else(|| polars_err!(InvalidOperation: "list values may not contain nulls"))
        })
        .collect()
}

fn build_tokenized_struct_series(
    name: PlSmallStr,
    raw_values: &[Option<String>],
    token_values: Vec<Option<Vec<String>>>,
    type_values: Vec<Option<Vec<String>>>,
    class_values: Vec<Option<Vec<String>>>,
) -> PolarsResult<Series> {
    let raw_series = StringChunked::from_iter_options(
        "raw_value".into(),
        raw_values.iter().map(|value| value.as_deref()),
    )
    .into_series();

    let token_series = build_string_list_series("tokens", token_values);
    let type_series = build_string_list_series("types", type_values);
    let class_series = build_string_list_series("classes", class_values);

    let fields = [raw_series, token_series, type_series, class_series];
    Ok(StructChunked::from_series(name, raw_values.len(), fields.iter())?.into_series())
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

fn init_extract_columns(
    capture_names: &[String],
    row_count: usize,
) -> Vec<(String, Vec<Option<String>>)> {
    capture_names
        .iter()
        .map(|name| (name.clone(), Vec::with_capacity(row_count)))
        .collect()
}

fn push_parse_output(
    field_columns: &mut [(String, Vec<Option<String>>)],
    complements: &mut Vec<Option<String>>,
    output: Option<ParseOutput>,
) {
    if let Some(output) = output {
        for (field_name, values) in field_columns.iter_mut() {
            values.push(output.fields.get(field_name).cloned());
        }
        complements.push(Some(output.complement));
    } else {
        for (_, values) in field_columns.iter_mut() {
            values.push(None);
        }
        complements.push(None);
    }
}

fn build_string_list_series(name: &str, rows: Vec<Option<Vec<String>>>) -> Series {
    let mut series = rows
        .into_iter()
        .map(|row| row.map(|values| Series::new(PlSmallStr::EMPTY, values)))
        .collect::<ListChunked>()
        .into_series();
    series.rename(name.into());
    series
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

    #[test]
    fn tokenize_helper_returns_struct_with_expected_fields() {
        let input = Series::new("address".into(), &[Some("123 MAIN ST"), None]);
        let output = tokenize_expr_impl(
            &[input],
            &TokenizeKwargs {
                model_path: fixture_model_path(),
            },
        )
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
        assert!(token_values.iter().any(|value| value == "123"));
        assert!(token_values.iter().any(|value| value == "MAIN"));
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
            &TokenizeKwargs {
                model_path: fixture_model_path(),
            },
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
                .capture_field_names("<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>")
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
