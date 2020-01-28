//! Contains XML tree serde Deserializer build on top of `Parser` from `parse` and `Deserializer`
//! from `de`.

use std::io::Read;
use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use tuple_utils::Prepend;

#[doc(hidden)]
pub use paste::item as paste_item; // Re-exported so we can use it in our macro. Is there better way?

use crate::de::{DeserializeError, Deserializer};
use crate::parser::{Event, Parser};

/// Utility that turns single-element tuple into that element and keeps multi-element tuples as
/// they are
pub trait DeTuple {
    /// The output type - a single element or >=2 element tuple
    type Output;

    /// Perform the transformation
    fn detuple(self) -> Self::Output;
}

impl<A> DeTuple for (A,) {
    type Output = A;
    fn detuple(self) -> Self::Output { self.0 }
}

macro_rules! detuple_impls {
    ($i:ident, $j:ident) => {
        detuple_impls!(impl $i, $j);
    };

    ($i:ident, $j:ident, $($r_i:ident),+) => {
        detuple_impls!(impl $i, $j, $($r_i),+);
        detuple_impls!($j, $($r_i),+);
    };

    (impl $($i:ident),+) => {
        impl<$($i),+> DeTuple for ($($i),+) {
            type Output = ($($i),+);
            fn detuple(self) -> Self::Output { self }
        }
    };
}

detuple_impls!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);


/// Like the try! macro (i.e. ? operator), but wraps the error in Some before returning.
macro_rules! try_some {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => return Some(Err(core::convert::From::from(err)))
        }
    }
}

/// Trait that determines whether given tag name matches some condition
pub trait TagMatcher {
    /// Does given `tag_name` match this matcher's condition?
    fn matches(&self, tag_name: &str) -> bool;
}

/// Matches tags against needle provided as `'static str`
#[derive(Debug, PartialEq)]
pub struct ExactTagMatch {
    needle: &'static str,
}

impl TagMatcher for ExactTagMatch {
    #[inline(always)]
    fn matches(&self, tag_name: &str) -> bool {
        self.needle == tag_name
    }
}

/// Matches all tags
#[derive(Debug, Default, PartialEq)]
pub struct AnyTagMatch {}

impl TagMatcher for AnyTagMatch {
    #[inline(always)]
    fn matches(&self, _tag_name: &str) -> bool {
        true
    }
}

/// Part of path for `TreeDeserializer`.
///
/// Enters matched element in a tree, nothing is deserialized.
///
/// You may want to use the `xml_path!` macro rather than constructing path manually.
#[derive(Debug)]
pub struct ElementEnter<M, N> {
    tag_matcher: M,
    next: N,

    entered: bool,
}

impl<M: Default, N: Default> Default for ElementEnter<M, N> {
    fn default() -> Self {
        Self {
            tag_matcher: M::default(),
            next: N::default(),
            entered: false,
        }
    }
}

impl<M, N> ElementEnter<M, N> {
    /// Create `ElementEnter` matching tag using given matcher
    pub fn new(tag_matcher: M, next: N) -> Self {
        Self { tag_matcher, next, entered: false }
    }
}

impl<N> ElementEnter<ExactTagMatch, N> {
    /// Create `ElementEnter` matching given tag
    pub fn tag(tag: &'static str, next: N) -> Self {
        Self::new(ExactTagMatch { needle: tag }, next)
    }
}

impl<N> ElementEnter<AnyTagMatch, N> {
    /// Create `ElementEnter` matching any tag
    pub fn any(next: N) -> Self {
        Self::new(AnyTagMatch {}, next)
    }
}

/// Part of path for `TreeDeserializer`.
///
/// Enters matched element in a tree. It's attributes are deserialized into given Deserializable
/// type and the deserializer proceeds with contained elements. The type must be `Clone` because it
/// may be returned multiple times if there are multiple contained elements.
///
/// You may want to use the `xml_path!` macro rather than constructing path manually.
#[derive(Debug)]
pub struct ElementEnterDeserialize<T, M, N> {
    tag_matcher: M,
    next: N,

    entered: Option<T>,
}

impl<T, M: Default, N: Default> Default for ElementEnterDeserialize<T, M, N> {
    fn default() -> Self {
        Self {
            tag_matcher: M::default(),
            next: N::default(),
            entered: None,
        }
    }
}

impl<T, M, N> ElementEnterDeserialize<T, M, N> {
    /// Create `ElementEnterDeserialize` matching tag using given matcher
    pub fn new(tag_matcher: M, next: N) -> Self {
        Self { tag_matcher, next, entered: None }
    }
}

impl<T, N> ElementEnterDeserialize<T, ExactTagMatch, N> {
    /// Create `ElementEnterDeserialize` matching given tag
    pub fn tag(tag: &'static str, next: N) -> Self {
        Self::new(ExactTagMatch { needle: tag }, next)
    }
}

impl<T, N> ElementEnterDeserialize<T, AnyTagMatch, N> {
    /// Create `ElementEnterDeserialize` matching any tag
    pub fn any(next: N) -> Self {
        Self::new(AnyTagMatch {}, next)
    }
}

/// Part of path for `TreeDeserializer`.
///
/// Deserializes matched element. This is the final component of every path. The matched element and
/// all nested elements are deserialized into given type.
///
/// You may want to use the `xml_path!` macro rather than constructing path manually.
#[derive(Debug, PartialEq)]
pub struct ElementDeserialize<T: DeserializeOwned, M> {
    tag_matcher: M,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned, M: Default> Default for ElementDeserialize<T, M> {
    fn default() -> Self {
        Self {
            tag_matcher: M::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T: DeserializeOwned, M> ElementDeserialize<T, M> {
    /// Create `ElementDeserialize` matching tag using given matcher
    pub fn new(tag_matcher: M) -> Self {
        Self { tag_matcher, _phantom: PhantomData }
    }
}

impl<T: DeserializeOwned> ElementDeserialize<T, ExactTagMatch> {
    /// Create `ElementDeserialize` matching given tag
    pub fn tag(tag: &'static str) -> Self {
        Self::new(ExactTagMatch { needle: tag })
    }
}

impl<T: DeserializeOwned> ElementDeserialize<T, AnyTagMatch> {
    /// Create `ElementDeserialize` matching any tag
    pub fn any() -> Self {
        Self::new(AnyTagMatch {})
    }
}

mod private {
    use serde::de::DeserializeOwned;

    pub trait Sealed {}

    impl<N, M> Sealed for super::ElementEnter<N, M> {}
    impl<T: DeserializeOwned, N, M> Sealed for super::ElementEnterDeserialize<T, N, M> {}
    impl<T: DeserializeOwned, M> Sealed for super::ElementDeserialize<T, M> {}
}

#[doc(hidden)]
pub trait XmlPath: private::Sealed {
    type Output: DeTuple;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, DeserializeError>>;
}

impl<M: TagMatcher, N: XmlPath> XmlPath for ElementEnter<M, N> {
    type Output = N::Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, DeserializeError>> {
        loop {
            if self.entered {
                if let Some(out) = self.next.go(parser) {
                    return Some(out);
                }
            }
            self.entered = false;

            let event = try_some!(parser.next());
            match event {
                Event::StartTag(tag_name) => {
                    let tag_name = try_some!(tag_name.to_str());
                    if self.tag_matcher.matches(tag_name.as_ref()) {
                        self.entered = true;
                    } else {
                        try_some!(parser.finish_tag(1));
                    }
                },
                Event::EndTagImmediate | Event::EndTag(_) | Event::Eof => {
                    return None;
                },
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => {}
            }
        }
    }
}

impl<T: DeserializeOwned + Clone, M: TagMatcher, N: XmlPath> XmlPath for ElementEnterDeserialize<T, M, N>
    where N::Output: Prepend<T>, <N::Output as Prepend<T>>::Output: DeTuple
{
    type Output = <N::Output as Prepend<T>>::Output;

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, DeserializeError>> {
        loop {
            if let Some(entered) = &self.entered {
                if let Some(out) = self.next.go(parser) {
                    return match out {
                        Ok(out) => {
                            // We clone one more times than necessary (the last remaining one will get dropped once
                            // the `next` returns `None`. It would be nice if we could avoid the last clone, but
                            // we would have to know that the underlying `next` really returned the last one.
                            Some(Ok(out.prepend((*entered).clone())))
                        }
                        Err(err) => Some(Err(err))
                    };
                }
            }
            self.entered = None;

            let event = try_some!(parser.next());
            match event {
                Event::StartTag(tag_name) => {
                    let tag_name = try_some!(tag_name.to_str());
                    if self.tag_matcher.matches(&tag_name) {
                        let opening_tag = tag_name.as_ref().into();
                        let mut des = Deserializer::new_inside_tag(parser, opening_tag, true);
                        self.entered = Some(try_some!(T::deserialize(&mut des)));
                    } else {
                        try_some!(parser.finish_tag(1));
                    }
                },
                Event::EndTagImmediate | Event::EndTag(_) => {
                    return None;
                },
                Event::Eof => {
                    return Some(Err(DeserializeError::UnexpectedEof));
                },
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => {}
            }
        }
    }
}

impl<T: DeserializeOwned, M: TagMatcher> XmlPath for ElementDeserialize<T, M> {
    type Output = (T,);

    fn go<R: Read>(&mut self, parser: &mut Parser<R>) -> Option<Result<Self::Output, DeserializeError>> {
        loop {
            let event = try_some!(parser.next());
            match event {
                Event::StartTag(tag_name) => {
                    let tag_name = try_some!(tag_name.to_str());
                    if self.tag_matcher.matches(tag_name.as_ref()) {
                        let opening_tag = tag_name.as_ref().into();
                        let mut des = Deserializer::new_inside_tag(parser, opening_tag, false);
                        return Some(Ok((try_some!(T::deserialize(&mut des)), )))
                    }
                },
                Event::EndTagImmediate | Event::EndTag(_) => {
                    return None;
                },
                Event::Eof => {
                    return Some(Err(DeserializeError::UnexpectedEof));
                },
                Event::StartTagDone | Event::AttributeName(_) | Event::AttributeValue(_) | Event::Text(_) => {}
            }
        }
    }
}

/// Deserializer for a sequence of elements in a tree.
///
/// See `xml_path` macro for easy way of constructing the path.
///
/// TreeDeserializer is an `Iterator` over the deserialized elements. If the xml path deserializes
/// exactly one type, the iterator will yield that type. If the xml path deserializes multiple
/// types (nested in a XML tree), the iterator will yield tuple of those types.
///
/// TODO: Example
pub struct TreeDeserializer<R: Read, N> {
    parser: Parser<R>,
    path: N,
    last_error_at: usize,
}

/// Type alias that helps to get the output type of TreeDeserializer
pub type TreeDeserializerOutput<N> = <<N as XmlPath>::Output as DeTuple>::Output;

impl<R: Read, N: XmlPath> TreeDeserializer<R, N> {
    /// Create new `TreeDeserializer` from given `Parser`.
    pub fn from_path(path: N, parser: Parser<R>) -> Self {
        Self {
            parser,
            path,
            last_error_at: std::usize::MAX,
        }
    }

    /// Create new `TreeDeserializer` from given IO `Read`.
    pub fn from_path_and_reader(path: N, reader: R) -> Self {
        Self::from_path(path, Parser::new(reader))
    }
}

impl<R: Read, N: XmlPath + Default> TreeDeserializer<R, N> {
    /// Create new `TreeDeserializer` from given `Parser`.
    pub fn new(parser: Parser<R>) -> Self {
        Self {
            parser,
            path: N::default(),
            last_error_at: std::usize::MAX,
        }
    }

    /// Create new `TreeDeserializer` from given IO `Read`.
    pub fn from_reader(reader: R) -> Self {
        Self::new(Parser::new(reader))
    }
}

impl<R: Read, N: XmlPath> Iterator for TreeDeserializer<R, N> where N::Output: DeTuple {
    type Item = Result<<N::Output as DeTuple>::Output, DeserializeError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.path.go(&mut self.parser) {
            Some(Ok(tuple)) => Some(Ok(tuple.detuple())),
            Some(Err(err)) => {
                let error_position = self.parser.last_position();
                if error_position == self.last_error_at {
                    // If we get another error at the same place, we consider the error
                    // unrecoverable and report no more elements
                    None
                } else {
                    self.last_error_at = error_position;
                    Some(Err(err))
                }
            }
            None => None,
        }
    }
}

/// Macro for easier construction of XML path.
///
/// This macro is alternative to building path by manually creating and nesting `ElementEnter`,
/// `ElementDeserialize` and `ElementEnterDeserialize`.
///
/// You can use `*` (not `"*"`!) to match any tag. Partial matching is currently not supported!
///
/// The last element must deserialize into a type!
///
/// # Example
///
/// ```no_run
/// # use serde_derive::Deserialize;
/// # #[derive(Clone, Deserialize)]
/// # struct Ccc {}
/// # #[derive(Deserialize)]
/// # struct Eee {}
/// # use rapid_xml::xml_path;
/// let path = xml_path!("aaa", *, "ccc" => Ccc, "ddd", * => Eee);
/// ```
///
/// This will enter tags `<aaa ...>`, inside enter any tag, inside enter and deserialize `<ccc ...>`
/// tag into `Ccc` type, enter `<ddd ...> tag tag and finaly deserialize any tag into `Eee` type.
///
/// The `TreeDeserializer` with this path will be `Iterator` over `Result<(Ccc, Eee), _>` type.
#[macro_export]
macro_rules! xml_path {
    // Tail rule - turns `"ccc" => Ccc` or `* => Ccc` expression into `ElementDeserialize`
    ($tag_name:literal => $t:ty) => {
        $crate::tree::ElementDeserialize::<$t, _>::tag($tag_name)
    };
    (* => $t:ty) => {
        $crate::tree::ElementDeserialize::<$t, _>::any()
    };

    // Tail rule - to inform user that the last expression must be `"ccc" => Ccc`, can't be just
    // `"ccc"`.
    ($tag_name:literal) => { $crate::xml_path!(*) };
    (*) => { compile_error!("Paths must end with `\"tag_name\" => Type` expression.") };

    // Recursive rule - turn `"ccc" => Ccc` or `* => Ccc` expression at beginning into `ElementEnterDeserialize`
    // and call ourselves recursively on the rest.
    ($tag_name:literal => $t:ty, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::ElementEnterDeserialize::<$t, _, _>::tag($tag_name,
            $crate::xml_path!($($r_tag_name $(=> $r_t)?),+)
        )
    };
    (* => $t:ty, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::ElementEnterDeserialize::<$t, _, _>::any(
            $crate::xml_path!($($r_tag_name $(=> $r_t)?),+)
        )
    };

    // Recursive rule - turn `"ccc"` or `*` expression at beginning into `ElementEnter` and call ourselves
    // recursively on the rest.
    ($tag_name:literal, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::ElementEnter::tag($tag_name,
            $crate::xml_path!($($r_tag_name $(=> $r_t)?),+)
        )
    };
    (*, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::ElementEnter::any(
            $crate::xml_path!($($r_tag_name $(=> $r_t)?),+)
        )
    };
}

/// Macro that expands to `type ... = ...` alias of XML path.
///
/// The XML path if fully encoded in the type, including the names of the tags to be matched.
///
/// The resulting type is `Default`-constructible. This allows you to create the `TreeDeserializer`
/// using the `new` and `from_reader` functions. This is useful if for example you need to store
/// the deserializer in an associated type.
///
/// # Example
///
/// ```no_run
/// # use serde_derive::Deserialize;
/// # #[derive(Clone, Deserialize)]
/// # struct Ccc {}
/// # #[derive(Deserialize)]
/// # struct Eee {}
/// # use rapid_xml::xml_path_type;
/// xml_path_type!(MyPath: "aaa", *, "ccc" => Ccc, "ddd", * => Eee);
/// # fn main() {}
/// ```
#[macro_export]
macro_rules! xml_path_type {
    // If we had const generics, implementing this would be quite easy - we would just have
    // something like the `ExactTagMatch` that takes the string as const generic parameter and here
    // we would just generate the type with it. But we don't have that in stable rust yet, so we use
    // more complicated technique: For every string in the input we generate a struct implementing
    // `TagMatcher` where the string ends up hardcoded inside the `matches` function. It is marked
    // as `#[inline(always)]`, so the final result should be the same as it would be after
    // monomorphization of the const generics struct.
    //
    // The generated structs are hidden inside a generated module to reduce chance of name
    // collisions. We use the `paste` crate to create names for the module and the structs
    // by concatenating the given type name with extra suffix.
    // The module ends up named "<type_name>__rapid_xml_generated_matchers", the structs inside end
    // up named "<Struct>Matcher<Suffix>", where `Struct` is the structure that the tag will be
    // deserialized into (if any, otherwise empty) and `Suffix` is string containing multiple
    // repeated 'A' characters. (Easiest form of a "counter".)

    // This is the entry point - this is what the user should call.
    // We create the basic structure - a module and type alias and call ourselves with @structs and
    // @types prefix to generate the structs and the type respectively.
    ($type_name:ident : $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            #[allow(non_snake_case)]
            #[doc(hidden)]
            mod [<$type_name __rapid_xml_generated_matchers>] {
                $crate::xml_path_type!(@structs A $($r_tag_name $(=> $r_t)?),+);
            }
        }

        type $type_name = $crate::xml_path_type!(@types $type_name A $($r_tag_name $(=> $r_t)?),+);
    };

    // == @structs ==
    // This part generates multiple structs implementing `TagMatcher` inside the module. One is
    // generated for each tag matches by string (none for tags matched by wildcard). It goes thru
    // the list recursively.

    // Tail of the recursion for string match, we generate the struct and the impl of TagMatcher.
    (@structs $suffix:ident $tag_name:literal $(=> $t:ty)?) => {
        $crate::tree::paste_item! {
            #[derive(Debug, Default, PartialEq)]
            pub struct [<$($t)? Matcher $suffix>] {}

            impl $crate::tree::TagMatcher for [<$($t)? Matcher $suffix>] {
                #[inline(always)]
                fn matches(&self, tag_name: &str) -> bool {
                    tag_name == $tag_name
                }
            }
        }
    };
    // Tail of the recursion for wildcard match, we generate nothing
    (@structs $suffix:ident * $(=> $t:ty)?) => {
        // nothing
    };

    // The next item in list is a string match, we generate the struct and impl (by calling
    // ourselves with the item alone - invoking the tail pattern) and then we call ourselves
    // recursively on the rest of the list.
    (@structs $suffix:ident $tag_name:literal $(=> $t:ty)?, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            $crate::xml_path_type!(@structs $suffix $tag_name $(=> $t)?);
            $crate::xml_path_type!(@structs [<$suffix A>] $($r_tag_name $(=> $r_t)?),*);
        }
    };

    // The next item in list is a wildcard match - we do nothing and call ourselves recursively on
    // the rest of the list.
    (@structs $suffix:ident * $(=> $t:ty)?, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            $crate::xml_path_type!(@structs [<$suffix A>] $($r_tag_name $(=> $r_t)?),+);
        }
    };

    // == @types ==
    // This part generates the type by nesting `ElementEnter`, `ElementEnterDeserialize` and
    // `ElementDeserialize`. They are build with matchers generated by the @structs part of with
    // `AnyTagMatch` matcher.

    // Tail rule - turns `"ccc" => Ccc` or `* => Ccc` expression into `ElementDeserialize`
    (@types $type_name:ident $suffix:ident $tag_name:literal => $t:ty) => {
        $crate::tree::paste_item! {
            $crate::tree::ElementDeserialize<$t, [<$type_name __rapid_xml_generated_matchers>]::[<$t Matcher $suffix>]>
        }
    };
    (@types $type_name:ident $suffix:ident * => $t:ty) => {
        $crate::tree::ElementDeserialize::<$t, $crate::tree::AnyTagMatch>
    };

    // Tail rule - to inform user that the last expression must be `"ccc" => Ccc`, can't be just
    // `"ccc"`.
    (@types $type_name:ident $suffix:ident $tag_name:literal) => { $crate::xml_path_type!(*) };
    (@types $type_name:ident $suffix:ident *) => { compile_error!("Paths must end with `\"tag_name\" => Type` expression.") };

    // Recursive rule - turn `"ccc" => Ccc` or `* => Ccc` expression at beginning into `ElementEnterDeserialize`
    // and call ourselves recursively on the rest.
    (@types $type_name:ident $suffix:ident $tag_name:literal => $t:ty, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            $crate::tree::ElementEnterDeserialize<$t, [<$type_name __rapid_xml_generated_matchers>]::[<$t Matcher $suffix>],
                $crate::xml_path_type!(@types $type_name [<$suffix A>] $($r_tag_name $(=> $r_t)?),+)
            >
        }
    };
    (@types $type_name:ident $suffix:ident * => $t:ty, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            $crate::tree::ElementEnterDeserialize::<$t, $crate::tree::AnyTagMatch,
                $crate::xml_path_type!(@types $type_name [<$suffix A>] $($r_tag_name $(=> $r_t)?),+)
            >
        }
    };

    // Recursive rule - turn `"ccc"` or `*` expression at beginning into `ElementEnter` and call ourselves
    // recursively on the rest.
    (@types $type_name:ident $suffix:ident $tag_name:literal, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            $crate::tree::ElementEnter<[<$type_name __rapid_xml_generated_matchers>]::[<Matcher $suffix>],
                $crate::xml_path_type!(@types $type_name [<$suffix A>] $($r_tag_name $(=> $r_t)?),+)
            >
        }
    };
    (@types $type_name:ident $suffix:ident *, $($r_tag_name:tt $(=> $r_t:ty)?),+) => {
        $crate::tree::paste_item! {
            $crate::tree::ElementEnter::<$crate::tree::AnyTagMatch,
                $crate::xml_path_type!(@types $type_name [<$suffix A>] $($r_tag_name $(=> $r_t)?),+)
            >
        }
    };
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use serde_derive::Deserialize;

    use super::*;

    #[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
    struct Bbb {
        n: u32,
    }

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Ccc {
        m: u32,
    }

    const SAMPLE_XML: &[u8] = br#"
            <root>
                <aaa>
                    <bbb n="1">
                        <ccc m="100"/>
                        <ccc m="200"/>
                    </bbb>
                    <xxx>Unknown tag</xxx>
                </aaa>
                <xxx>Unknown tag</xxx>
                <aaa>
                    <bbb n="99">Matched tag without anything nested</bbb>
                    <bbb n="2">
                        <ccc><m>300</m></ccc>
                        <ccc><m>400</m></ccc>
                    </bbb>
                </aaa>
                <aaa2>
                    <bbb n="3">
                        <ccc m="500"/>
                    </bbb>
                </aaa2>
            </root>
        "#;

    const SAMPLE_XML_ERRORS: &[u8] = br#"
            <root>
                <aaa>
                    <bbb n="1">
                        <ccc m="100"/>
                        <ccc/>
                        <ccc m="200"/>
                        <ccc m=250/>
                    </bbb>
                    <xxx>Unknown tag</xxx>
                </aaa>
                <xxx>Unknown tag</xxx>
                <aaa>
                    <bbb n="99">Matched tag without anything nested</bbb>
                    <bbb n="2">
                        <ccc><m>300</m></ccc>
                        <ccc><m>asdf</m></ccc>
                    </bbb>
                </aaa>
            </root>
        "#;

    #[test]
    fn basic() {
        let path = xml_path!("root", "aaa", "bbb", "ccc" => Ccc);

        let mut des = TreeDeserializer::from_path_and_reader(path, Cursor::new(&SAMPLE_XML[..]));

        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 100 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 200 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 300 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 400 });
        assert!(des.next().is_none());
    }

    #[test]
    fn wildcard() {
        let path = xml_path!("root", *, "bbb", "ccc" => Ccc);

        let mut des = TreeDeserializer::from_path_and_reader(path, Cursor::new(&SAMPLE_XML[..]));

        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 100 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 200 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 300 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 400 });
        assert_eq!(des.next().unwrap().unwrap(), Ccc { m: 500 });
        assert!(des.next().is_none());
    }

    #[test]
    fn multiple_elements() {
        let path = xml_path!("root", "aaa", "bbb" => Bbb, "ccc" => Ccc);

        let mut des = TreeDeserializer::from_path_and_reader(path, Cursor::new(&SAMPLE_XML[..]));

        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 100 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 200 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 300 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 400 }));
        assert!(des.next().is_none());
    }

    xml_path_type!(MyPath: "root", "aaa", "bbb" => Bbb, "ccc" => Ccc);

    #[test]
    fn multiple_elements_with_type() {
        let mut des = TreeDeserializer::<_, MyPath>::from_reader(Cursor::new(&SAMPLE_XML[..]));

        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 100 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 200 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 300 }));
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 400 }));
        assert!(des.next().is_none());
    }

    #[test]
    fn with_errors() {
        let path = xml_path!("root", "aaa", "bbb" => Bbb, "ccc" => Ccc);

        let mut des = TreeDeserializer::from_path_and_reader(path, Cursor::new(&SAMPLE_XML_ERRORS[..]));

        // TODO: The actual output may still change if we learn to recover from some errors better.

        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 100 }));
        assert!(des.next().unwrap().is_err());
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 1 }, Ccc { m: 200 }));
        assert!(des.next().unwrap().is_err()); // Bad attribute value
        assert!(des.next().unwrap().is_err()); // Bad attribute name (rest of the bad value)
        assert_eq!(des.next().unwrap().unwrap(), (Bbb { n: 2 }, Ccc { m: 300 }));
        assert!(des.next().unwrap().is_err());
        assert!(des.next().is_none());
    }
}
