//! Turns the documentation of the settings into a table the applications can show.
//!
//! The doc comments in `src/configuration/settings.rs` are the single source of
//! truth for what each setting does. This script copies them into a generated
//! table so the command-line and desktop applications explain every setting
//! with exactly the words written next to its definition.

use std::collections::BTreeMap;
use std::path::Path;

const SETTINGS_SOURCE_PATH: &str = "src/configuration/settings.rs";
const ROOT_STRUCT_NAME: &str = "RuntimeSettings";
const GENERATED_FILE_NAME: &str = "setting_descriptions.rs";
const DOC_COMMENT_MARKER: &str = "///";
const STRUCT_OPENING: &str = "pub struct ";
const FIELD_OPENING: &str = "pub ";

/// One documented field of a settings struct.
struct DocumentedField {
    name: String,
    type_name: String,
    description: String,
}

fn main() {
    println!("cargo:rerun-if-changed={SETTINGS_SOURCE_PATH}");
    println!("cargo:rerun-if-changed=build.rs");

    let source = std::fs::read_to_string(SETTINGS_SOURCE_PATH)
        .expect("the settings source file should be readable");
    let fields_by_struct = collect_documented_fields(&source);
    let descriptions = describe_setting_paths(&fields_by_struct);

    let output_directory = std::env::var("OUT_DIR").expect("cargo should set OUT_DIR");
    std::fs::write(
        Path::new(&output_directory).join(GENERATED_FILE_NAME),
        render_description_table(&descriptions),
    )
    .expect("the generated description table should be writable");
}

/// Reads every `pub field: Type,` with the doc comment above it, grouped by struct.
fn collect_documented_fields(source: &str) -> BTreeMap<String, Vec<DocumentedField>> {
    let mut fields_by_struct: BTreeMap<String, Vec<DocumentedField>> = BTreeMap::new();
    let mut current_struct_name: Option<String> = None;
    let mut pending_documentation: Vec<&str> = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(documentation) = trimmed.strip_prefix(DOC_COMMENT_MARKER) {
            pending_documentation.push(documentation.trim());
            continue;
        }
        if let Some(declaration) = trimmed.strip_prefix(STRUCT_OPENING) {
            current_struct_name = declaration.split_whitespace().next().map(str::to_string);
        }
        let is_struct_end = line.starts_with('}');
        if is_struct_end {
            current_struct_name = None;
        }
        if let (Some(struct_name), Some(field)) = (
            &current_struct_name,
            parse_field(trimmed, &pending_documentation),
        ) {
            fields_by_struct
                .entry(struct_name.clone())
                .or_default()
                .push(field);
        }
        pending_documentation.clear();
    }
    fields_by_struct
}

/// Parses `pub name: Type,`; anything else is not a field.
fn parse_field(trimmed_line: &str, documentation: &[&str]) -> Option<DocumentedField> {
    let declaration = trimmed_line
        .strip_prefix(FIELD_OPENING)?
        .strip_suffix(',')?;
    let (name, type_name) = declaration.split_once(':')?;
    Some(DocumentedField {
        name: name.trim().to_string(),
        type_name: type_name.trim().to_string(),
        description: join_documentation(documentation),
    })
}

/// Joins doc comment lines into one paragraph, skipping the blank separators.
fn join_documentation(documentation: &[&str]) -> String {
    documentation
        .iter()
        .filter(|line| !line.is_empty())
        .copied()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Pairs every group and every `group.field` path with its description.
fn describe_setting_paths(
    fields_by_struct: &BTreeMap<String, Vec<DocumentedField>>,
) -> Vec<(String, String)> {
    let groups = fields_by_struct
        .get(ROOT_STRUCT_NAME)
        .expect("the settings source should define the root settings struct");

    let mut descriptions = Vec::new();
    for group in groups {
        descriptions.push((group.name.clone(), group.description.clone()));
        let group_fields = fields_by_struct
            .get(&group.type_name)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for field in group_fields {
            let path = format!("{}.{}", group.name, field.name);
            descriptions.push((path, field.description.clone()));
        }
    }
    descriptions
}

fn render_description_table(descriptions: &[(String, String)]) -> String {
    let rows: String = descriptions
        .iter()
        .map(|(path, description)| format!("    ({path:?}, {description:?}),\n"))
        .collect();
    format!(
        "/// Setting and group descriptions, copied from the doc comments on the settings.\n\
         const SETTING_DESCRIPTIONS: &[(&str, &str)] = &[\n{rows}];\n"
    )
}
