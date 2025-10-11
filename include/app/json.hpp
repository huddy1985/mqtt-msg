#pragma once

#include <cctype>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace app {

class Json {
public:
    using object_t = std::map<std::string, Json>;
    using array_t = std::vector<Json>;
    using value_t = std::variant<std::nullptr_t, bool, double, std::string, object_t, array_t>;

    Json() : value_(nullptr) {}
    Json(std::nullptr_t) : value_(nullptr) {}
    Json(bool v) : value_(v) {}
    Json(double v) : value_(v) {}
    Json(std::string v) : value_(std::move(v)) {}
    Json(object_t v) : value_(std::move(v)) {}
    Json(array_t v) : value_(std::move(v)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
    bool is_boolean() const { return std::holds_alternative<bool>(value_); }
    bool is_number() const { return std::holds_alternative<double>(value_); }
    bool is_string() const { return std::holds_alternative<std::string>(value_); }
    bool is_object() const { return std::holds_alternative<object_t>(value_); }
    bool is_array() const { return std::holds_alternative<array_t>(value_); }

    const value_t &value() const { return value_; }
    value_t &value() { return value_; }

    const object_t &as_object() const { return std::get<object_t>(value_); }
    object_t &as_object() { return std::get<object_t>(value_); }

    const array_t &as_array() const { return std::get<array_t>(value_); }
    array_t &as_array() { return std::get<array_t>(value_); }

    const std::string &as_string() const { return std::get<std::string>(value_); }
    std::string &as_string() { return std::get<std::string>(value_); }

    double as_number() const { return std::get<double>(value_); }

    bool as_bool() const { return std::get<bool>(value_); }

    const Json &operator[](const std::string &key) const {
        const auto &obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::out_of_range("key not found: " + key);
        }
        return it->second;
    }

    Json &operator[](const std::string &key) {
        auto &obj = as_object();
        return obj[key];
    }

    const Json &operator[](std::size_t index) const {
        const auto &arr = as_array();
        if (index >= arr.size()) {
            throw std::out_of_range("array index out of range");
        }
        return arr[index];
    }

    Json &operator[](std::size_t index) {
        auto &arr = as_array();
        if (index >= arr.size()) {
            throw std::out_of_range("array index out of range");
        }
        return arr[index];
    }

    bool contains(const std::string &key) const {
        if (!is_object()) {
            return false;
        }
        const auto &obj = as_object();
        return obj.find(key) != obj.end();
    }

    std::string dump(int indent = -1) const {
        std::string out;
        dump_impl(out, indent, 0);
        return out;
    }

    static Json parse(const std::string &text);

private:
    value_t value_;

    void dump_impl(std::string &out, int indent, int depth) const;
};

class JsonParser {
public:
    explicit JsonParser(const std::string &text) : s_(text) {}

    Json parse();

private:
    const std::string &s_;
    std::size_t pos_{0};

    void skip_ws();
    char peek() const;
    bool consume(char expected);

    Json parse_value();
    Json parse_object();
    Json parse_array();
    Json parse_string();
    Json parse_bool();
    Json parse_null();
    Json parse_number();
};

inline Json Json::parse(const std::string &text) {
    JsonParser parser(text);
    Json result = parser.parse();
    return result;
}

inline void Json::dump_impl(std::string &out, int indent, int depth) const {
    if (is_null()) {
        out += "null";
        return;
    }
    if (is_boolean()) {
        out += as_bool() ? "true" : "false";
        return;
    }
    if (is_number()) {
        out += std::to_string(as_number());
        return;
    }
    if (is_string()) {
        out.push_back('"');
        for (char c : as_string()) {
            switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
            }
        }
        out.push_back('"');
        return;
    }
    if (is_array()) {
        out.push_back('[');
        const auto &arr = as_array();
        for (std::size_t i = 0; i < arr.size(); ++i) {
            if (i > 0) {
                out.push_back(',');
            }
            if (indent >= 0) {
                out.push_back('\n');
                out.append((depth + 1) * indent, ' ');
            }
            arr[i].dump_impl(out, indent, depth + 1);
        }
        if (!arr.empty() && indent >= 0) {
            out.push_back('\n');
            out.append(depth * indent, ' ');
        }
        out.push_back(']');
        return;
    }
    if (is_object()) {
        out.push_back('{');
        const auto &obj = as_object();
        std::size_t count = 0;
        for (const auto &kv : obj) {
            if (count++ > 0) {
                out.push_back(',');
            }
            if (indent >= 0) {
                out.push_back('\n');
                out.append((depth + 1) * indent, ' ');
            }
            out.push_back('"');
            out += kv.first;
            out.push_back('"');
            out.push_back(':');
            if (indent >= 0) {
                out.push_back(' ');
            }
            kv.second.dump_impl(out, indent, depth + 1);
        }
        if (!obj.empty() && indent >= 0) {
            out.push_back('\n');
            out.append(depth * indent, ' ');
        }
        out.push_back('}');
        return;
    }
}

inline void JsonParser::skip_ws() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
        ++pos_;
    }
}

inline char JsonParser::peek() const {
    if (pos_ >= s_.size()) {
        return '\0';
    }
    return s_[pos_];
}

inline bool JsonParser::consume(char expected) {
    skip_ws();
    if (peek() != expected) {
        return false;
    }
    ++pos_;
    return true;
}

inline Json JsonParser::parse() {
    Json value = parse_value();
    skip_ws();
    if (pos_ != s_.size()) {
        throw std::runtime_error("unexpected trailing characters in JSON");
    }
    return value;
}

inline Json JsonParser::parse_value() {
    skip_ws();
    char c = peek();
    if (c == '"') {
        return parse_string();
    }
    if (c == '{') {
        return parse_object();
    }
    if (c == '[') {
        return parse_array();
    }
    if (c == 't' || c == 'f') {
        return parse_bool();
    }
    if (c == 'n') {
        return parse_null();
    }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
        return parse_number();
    }
    throw std::runtime_error("invalid JSON value");
}

inline Json JsonParser::parse_object() {
    if (!consume('{')) {
        throw std::runtime_error("expected '{'");
    }
    Json::object_t obj;
    skip_ws();
    if (consume('}')) {
        return obj;
    }
    while (true) {
        skip_ws();
        if (peek() != '"') {
            throw std::runtime_error("expected string key");
        }
        std::string key = parse_string().as_string();
        skip_ws();
        if (!consume(':')) {
            throw std::runtime_error("expected ':' after key");
        }
        Json value = parse_value();
        obj.emplace(std::move(key), std::move(value));
        skip_ws();
        if (consume('}')) {
            break;
        }
        if (!consume(',')) {
            throw std::runtime_error("expected ',' in object");
        }
    }
    return obj;
}

inline Json JsonParser::parse_array() {
    if (!consume('[')) {
        throw std::runtime_error("expected '['");
    }
    Json::array_t arr;
    skip_ws();
    if (consume(']')) {
        return arr;
    }
    while (true) {
        arr.emplace_back(parse_value());
        skip_ws();
        if (consume(']')) {
            break;
        }
        if (!consume(',')) {
            throw std::runtime_error("expected ',' in array");
        }
    }
    return arr;
}

inline Json JsonParser::parse_string() {
    if (!consume('"')) {
        throw std::runtime_error("expected string opening quote");
    }
    std::string out;
    while (pos_ < s_.size()) {
        char c = s_[pos_++];
        if (c == '"') {
            break;
        }
        if (c == '\\') {
            if (pos_ >= s_.size()) {
                throw std::runtime_error("invalid escape sequence");
            }
            char esc = s_[pos_++];
            switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            default:
                throw std::runtime_error("unsupported escape sequence");
            }
        } else {
            out.push_back(c);
        }
    }
    return Json(std::move(out));
}

inline Json JsonParser::parse_bool() {
    if (s_.compare(pos_, 4, "true") == 0) {
        pos_ += 4;
        return Json(true);
    }
    if (s_.compare(pos_, 5, "false") == 0) {
        pos_ += 5;
        return Json(false);
    }
    throw std::runtime_error("invalid boolean literal");
}

inline Json JsonParser::parse_null() {
    if (s_.compare(pos_, 4, "null") != 0) {
        throw std::runtime_error("invalid null literal");
    }
    pos_ += 4;
    return Json(nullptr);
}

inline Json JsonParser::parse_number() {
    std::size_t start = pos_;
    if (s_[pos_] == '-') {
        ++pos_;
    }
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
        ++pos_;
    }
    if (pos_ < s_.size() && s_[pos_] == '.') {
        ++pos_;
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }
    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
        ++pos_;
        if (s_[pos_] == '+' || s_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }
    double value = std::stod(s_.substr(start, pos_ - start));
    return Json(value);
}

} // namespace app

