use dojo::meta::{FieldLayout, Layout};
use dojo::utils::{find_field_layout, find_model_field_layout};

#[test]
fn test_find_layout_when_exists() {
    let layouts = [
        FieldLayout { selector: 'one', layout: Layout::Fixed([1].span()) },
        FieldLayout { selector: 'two', layout: Layout::Fixed([2].span()) },
        FieldLayout { selector: 'three', layout: Layout::Fixed([3].span()) },
    ]
        .span();

    let res = find_field_layout('two', layouts);
    assert(res.is_some(), 'layout not found');
    let res = res.unwrap();
    assert(res == Layout::Fixed([2].span()), 'bad layout');
}

#[test]
fn test_find_layout_fails_when_not_exists() {
    let layouts = [
        FieldLayout { selector: 'one', layout: Layout::Fixed([1].span()) },
        FieldLayout { selector: 'two', layout: Layout::Fixed([2].span()) },
        FieldLayout { selector: 'three', layout: Layout::Fixed([3].span()) },
    ]
        .span();

    let res = find_field_layout('four', layouts);
    assert(res.is_none(), 'layout found');
}

#[test]
fn test_find_model_layout_when_exists() {
    let model_layout = Layout::Struct(
        [
            FieldLayout { selector: 'one', layout: Layout::Fixed([1].span()) },
            FieldLayout { selector: 'two', layout: Layout::Fixed([2].span()) },
            FieldLayout { selector: 'three', layout: Layout::Fixed([3].span()) },
        ]
            .span(),
    );

    let res = find_model_field_layout(model_layout, 'two');
    assert(res.is_some(), 'layout not found');
    let res = res.unwrap();
    assert(res == Layout::Fixed([2].span()), 'bad layout');
}

#[test]
fn test_find_model_layout_fails_when_not_exists() {
    let model_layout = Layout::Struct(
        [
            FieldLayout { selector: 'one', layout: Layout::Fixed([1].span()) },
            FieldLayout { selector: 'two', layout: Layout::Fixed([2].span()) },
            FieldLayout { selector: 'three', layout: Layout::Fixed([3].span()) },
        ]
            .span(),
    );

    let res = find_model_field_layout(model_layout, 'four');
    assert(res.is_none(), 'layout found');
}

#[test]
fn test_find_model_layout_returns_none_for_non_struct_layout() {
    let res = find_model_field_layout(Layout::Fixed([].span()), 'one');
    assert(res.is_none(), 'should be None for Fixed');
}
