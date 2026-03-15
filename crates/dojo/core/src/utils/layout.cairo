use dojo::meta::{FieldLayout, Layout};

/// find a field with its selector in a list of layouts
pub fn find_field_layout(
    field_selector: felt252, field_layouts: Span<FieldLayout>,
) -> Option<Layout> {
    for field_layout in field_layouts {
        if field_selector == *field_layout.selector {
            return Option::Some(*field_layout.layout);
        }
    }

    Option::None
}

/// Find the layout of a model field based on its selector.
///
/// Only `Layout::Struct` models have named fields.  For `Layout::Fixed`
/// and other layout variants the function returns `None` because there is
/// no field-level structure to look up.
///
/// # Arguments
///
/// * `model_layout` - The full model layout.
/// * `member_selector` - The model field selector.
///
/// # Returns
/// Some(Layout) if the field layout has been found, None otherwise.
pub fn find_model_field_layout(model_layout: Layout, member_selector: felt252) -> Option<Layout> {
    match model_layout {
        Layout::Struct(field_layouts) => { find_field_layout(member_selector, field_layouts) },
        _ => Option::None,
    }
}
