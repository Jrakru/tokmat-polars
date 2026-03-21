#![doc = include_str!("../README.md")]

//! Runtime crate for the wanParser Polars integration layer.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use tokmat::extractor::{Extractor, MatchMode};
use tokmat::tel::CompiledPattern;
use tokmat::token_model::TokenModel;
use tokmat::tokenizer::{TokenizedResult, tokenize_with_model};

static CONTEXT_CACHE: LazyLock<Mutex<HashMap<String, Arc<ModelContext>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct ModelContext {
    model: TokenModel,
    extractor: Extractor,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizeKwargs {
    model_path: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ExtractKwargs {
    model_path: String,
    pattern: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizedRow {
    raw_value: String,
    tokens: Vec<String>,
    classes: Vec<String>,
}

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func=tokenize_output_type)]
fn tokenize_expr(inputs: &[Series], kwargs: TokenizeKwargs) -> PolarsResult<Series> {
    tokenize_expr_impl(inputs, &kwargs)
}

#[allow(clippy::needless_pass_by_value)]
#[polars_expr(output_type_func_with_kwargs=extract_output_type)]
fn extract_expr(inputs: &[Series], kwargs: ExtractKwargs) -> PolarsResult<Series> {
    extract_expr_impl(inputs, &kwargs)
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
    let rows = string_rows(input)?;

    let tokenized_rows = rows
        .iter()
        .map(|value| {
            value
                .as_deref()
                .map(|raw_value| tokenize_with_model(raw_value, &context.model))
        })
        .collect::<Vec<_>>();

    build_tokenized_struct_series(input.name().clone(), &rows, &tokenized_rows)
}

fn extract_expr_impl(inputs: &[Series], kwargs: &ExtractKwargs) -> PolarsResult<Series> {
    let input = single_input(inputs, "extract_expr")?;
    let context = get_or_load_context(&kwargs.model_path)?;
    let capture_names = capture_field_names_from_pattern(&kwargs.pattern)?;

    let parsed_rows = match input.dtype() {
        DataType::String => extract_from_string_series(input, &context, kwargs)?,
        DataType::Struct(_) => extract_from_tokenized_series(input, &context, kwargs)?,
        dtype => {
            polars_bail!(
                InvalidOperation:
                "extract_expr expected a String or tokenized Struct column, got {:?}",
                dtype
            )
        }
    };

    build_extract_struct_series(input.name().clone(), &capture_names, &parsed_rows)
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

fn string_rows(series: &Series) -> PolarsResult<Vec<Option<String>>> {
    Ok(series
        .str()?
        .into_iter()
        .map(|value| value.map(ToString::to_string))
        .collect())
}

fn extract_from_string_series(
    input: &Series,
    context: &ModelContext,
    kwargs: &ExtractKwargs,
) -> PolarsResult<Vec<Option<HashMap<String, String>>>> {
    let rows = string_rows(input)?;
    rows.into_iter()
        .map(|raw_value| match raw_value {
            Some(raw_value) => {
                let tokenized = tokenize_with_model(&raw_value, &context.model);
                let parsed = context
                    .extractor
                    .parse_tokens(
                        &raw_value,
                        &tokenized.tokens,
                        &tokenized.classes,
                        &kwargs.pattern,
                        MatchMode::Whole,
                    )
                    .map_err(|error| {
                        polars_err!(
                            ComputeError:
                            "failed to extract TEL pattern '{}': {}",
                            kwargs.pattern,
                            error
                        )
                    })?;
                Ok(Some(parse_output_to_row(parsed)))
            }
            None => Ok(None),
        })
        .collect()
}

fn extract_from_tokenized_series(
    input: &Series,
    context: &ModelContext,
    kwargs: &ExtractKwargs,
) -> PolarsResult<Vec<Option<HashMap<String, String>>>> {
    let rows = tokenized_rows_from_struct(input)?;
    rows.into_iter()
        .map(|row| match row {
            Some(row) => {
                let parsed = context
                    .extractor
                    .parse_tokens(
                        &row.raw_value,
                        &row.tokens,
                        &row.classes,
                        &kwargs.pattern,
                        MatchMode::Whole,
                    )
                    .map_err(|error| {
                        polars_err!(
                            ComputeError:
                            "failed to extract TEL pattern '{}': {}",
                            kwargs.pattern,
                            error
                        )
                    })?;
                Ok(Some(parse_output_to_row(parsed)))
            }
            None => Ok(None),
        })
        .collect()
}

fn parse_output_to_row(output: tokmat::extractor::ParseOutput) -> HashMap<String, String> {
    let mut row = output.fields;
    row.insert("complement".to_string(), output.complement);
    row
}

fn tokenized_rows_from_struct(series: &Series) -> PolarsResult<Vec<Option<TokenizedRow>>> {
    let struct_chunked = series.struct_()?;
    let fields = struct_chunked.fields_as_series();
    let field_map = fields
        .into_iter()
        .map(|field| (field.name().to_string(), field))
        .collect::<HashMap<_, _>>();

    let raw_field = field_map.get("raw_value").ok_or_else(|| {
        polars_err!(
            InvalidOperation:
            "tokenized struct is missing required 'raw_value' field"
        )
    })?;
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

    let raw_values = string_rows(raw_field)?;
    let token_lists = tokens_field.list()?.into_iter().collect::<Vec<_>>();
    let class_lists = classes_field.list()?.into_iter().collect::<Vec<_>>();

    let mut rows = Vec::with_capacity(series.len());
    for index in 0..series.len() {
        match (&raw_values[index], &token_lists[index], &class_lists[index]) {
            (Some(raw_value), Some(tokens), Some(classes)) => rows.push(Some(TokenizedRow {
                raw_value: raw_value.clone(),
                tokens: list_series_to_strings(tokens)?,
                classes: list_series_to_strings(classes)?,
            })),
            (None, None, None) => rows.push(None),
            _ => {
                polars_bail!(
                    InvalidOperation:
                    "tokenized struct row {} has inconsistent nullability across fields",
                    index
                )
            }
        }
    }

    Ok(rows)
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
    tokenized_rows: &[Option<TokenizedResult>],
) -> PolarsResult<Series> {
    let raw_series = StringChunked::from_iter_options(
        "raw_value".into(),
        raw_values.iter().map(|value| value.as_deref()),
    )
    .into_series();

    let token_series = build_string_list_series(
        "tokens",
        tokenized_rows
            .iter()
            .map(|row| row.as_ref().map(|tokenized| tokenized.tokens.clone())),
    );
    let type_series = build_string_list_series(
        "types",
        tokenized_rows
            .iter()
            .map(|row| row.as_ref().map(|tokenized| tokenized.types.clone())),
    );
    let class_series = build_string_list_series(
        "classes",
        tokenized_rows
            .iter()
            .map(|row| row.as_ref().map(|tokenized| tokenized.classes.clone())),
    );

    let fields = [raw_series, token_series, type_series, class_series];
    Ok(StructChunked::from_series(name, raw_values.len(), fields.iter())?.into_series())
}

fn build_extract_struct_series(
    name: PlSmallStr,
    capture_names: &[String],
    rows: &[Option<HashMap<String, String>>],
) -> PolarsResult<Series> {
    let mut field_series = capture_names
        .iter()
        .map(|field_name| {
            StringChunked::from_iter_options(
                field_name.clone().into(),
                rows.iter().map(|row| {
                    row.as_ref()
                        .and_then(|values| values.get(field_name))
                        .map(String::as_str)
                }),
            )
            .into_series()
        })
        .collect::<Vec<_>>();

    field_series.push(
        StringChunked::from_iter_options(
            "complement".into(),
            rows.iter().map(|row| {
                row.as_ref()
                    .and_then(|values| values.get("complement"))
                    .map(String::as_str)
            }),
        )
        .into_series(),
    );

    Ok(StructChunked::from_series(name, rows.len(), field_series.iter())?.into_series())
}

fn build_string_list_series(name: &str, rows: impl Iterator<Item = Option<Vec<String>>>) -> Series {
    let mut series = rows
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
            &ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
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
            &ExtractKwargs {
                model_path: fixture_model_path(),
                pattern: "<<CIVIC#>> <<STREET@+>> <<TYPE::STREETTYPE>>".to_string(),
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
}
