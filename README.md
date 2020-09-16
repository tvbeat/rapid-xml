# Rapid XML

Rapid XML is library for parsing XML. It focuses on performance and deserialization with serde.

This library provides 3 ways of reading XML, each building on top of the previous one:

 - `Parser`: Low-level parser that quickly turns a stream of bytes from IO `Read` into a stream
             of  events, such as "start tag", "attribute name", "attribute value", "end tag", ...
 - `Deserializer`: Consumes events from `Parser` and constructs any type that is deserializable
                  by serde.
 - `TreeDeserializer`: Deserializes sequences of (optionally nested) types from XML trees.
 