#pragma once

#include <cctype>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace simplejson {

class JsonValue {
public:
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    using Variant = std::variant<std::nullptr_t, bool, double, std::string, Object, Array>;

    JsonValue() : data_(nullptr) {}
    JsonValue(std::nullptr_t) : data_(nullptr) {}
    JsonValue(bool value) : data_(value) {}
    JsonValue(int value) : data_(static_cast<double>(value)) {}
    JsonValue(double value) : data_(value) {}
    JsonValue(const char* value) : data_(std::string(value)) {}
    JsonValue(const std::string& value) : data_(value) {}
    JsonValue(std::string&& value) : data_(std::move(value)) {}
    JsonValue(const Object& value) : data_(value) {}
    JsonValue(Object&& value) : data_(std::move(value)) {}
    JsonValue(const Array& value) : data_(value) {}
    JsonValue(Array&& value) : data_(std::move(value)) {}

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(data_); }
    bool isBool() const { return std::holds_alternative<bool>(data_); }
    bool isNumber() const { return std::holds_alternative<double>(data_); }
    bool isString() const { return std::holds_alternative<std::string>(data_); }
    bool isObject() const { return std::holds_alternative<Object>(data_); }
    bool isArray() const { return std::holds_alternative<Array>(data_); }

    const JsonValue &operator[](const std::string &key) const {
        const auto &obj = asObject();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::out_of_range("key not found: " + key);
        }
        return it->second;
    }

    JsonValue &operator[](const std::string &key) {
        auto &obj = asObject();
        return obj[key];
    }

    const JsonValue &operator[](std::size_t index) const {
        const auto &arr = asArray();
        if (index >= arr.size()) {
            throw std::out_of_range("array index out of range");
        }
        return arr[index];
    }

    JsonValue &operator[](std::size_t index) {
        auto &arr = asArray();
        if (index >= arr.size()) {
            throw std::out_of_range("array index out of range");
        }
        return arr[index];
    }

    bool asBool(bool default_value = false) const {
        if (isBool()) {
            return std::get<bool>(data_);
        }
        return default_value;
    }

    double asNumber(double default_value = 0.0) const {
        if (isNumber()) {
            return std::get<double>(data_);
        }
        return default_value;
    }

    const std::string& asString() const {
        if (!isString()) {
            throw std::runtime_error("JSON value is not a string");
        }
        return std::get<std::string>(data_);
    }

    const Object& asObject() const {
        if (!isObject()) {
            throw std::runtime_error("JSON value is not an object");
        }
        return std::get<Object>(data_);
    }

    Object& asObject() {
        if (!isObject()) {
            data_ = Object{};
        }
        return std::get<Object>(data_);
    }

    const Array& asArray() const {
        if (!isArray()) {
            throw std::runtime_error("JSON value is not an array");
        }
        return std::get<Array>(data_);
    }

    Array& asArray() {
        if (!isArray()) {
            data_ = Array{};
        }
        return std::get<Array>(data_);
    }

    bool contains(const std::string& key) const {
        if (!isObject()) {
            return false;
        }
        const auto& obj = std::get<Object>(data_);
        return obj.find(key) != obj.end();
    }

    const JsonValue& at(const std::string& key) const {
        const auto& obj = asObject();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("Missing key: " + key);
        }
        return it->second;
    }

    std::string getString(const std::string& key, const std::string& default_value = "") const {
        if (!contains(key)) {
            return default_value;
        }
        const auto& value = at(key);
        if (!value.isString()) {
            throw std::runtime_error("Expected string for key: " + key);
        }
        return value.asString();
    }

    double getNumber(const std::string& key, double default_value = 0.0) const {
        if (!contains(key)) {
            return default_value;
        }
        const auto& value = at(key);
        if (!value.isNumber()) {
            throw std::runtime_error("Expected number for key: " + key);
        }
        return value.asNumber();
    }

    bool getBool(const std::string& key, bool default_value = false) const {
        if (!contains(key)) {
            return default_value;
        }
        const auto& value = at(key);
        if (!value.isBool()) {
            throw std::runtime_error("Expected bool for key: " + key);
        }
        return value.asBool();
    }

    const Array& getArray(const std::string& key) const {
        return at(key).asArray();
    }

    std::string dump(int indent = -1) const {
        std::ostringstream oss;
        dumpInternal(oss, indent, 0);
        return oss.str();
    }

    const Variant& data() const { return data_; }

private:
    Variant data_;

    static std::string escapeString(const std::string& value) {
        std::ostringstream oss;
        oss << '"';
        for (char ch : value) {
            switch (ch) {
                case '\\': oss << "\\\\"; break;
                case '\"': oss << "\\\""; break;
                case '\b': oss << "\\b"; break;
                case '\f': oss << "\\f"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default:
                    if (static_cast<unsigned char>(ch) < 0x20) {
                        oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(static_cast<unsigned char>(ch));
                        oss << std::dec;
                    } else {
                        oss << ch;
                    }
            }
        }
        oss << '"';
        return oss.str();
    }

    void dumpInternal(std::ostringstream& oss, int indent, int depth) const {
        if (isNull()) {
            oss << "null";
        } else if (isBool()) {
            oss << (asBool() ? "true" : "false");
        } else if (isNumber()) {
            oss << std::get<double>(data_);
        } else if (isString()) {
            oss << escapeString(std::get<std::string>(data_));
        } else if (isArray()) {
            const auto& arr = std::get<Array>(data_);
            oss << '[';
            if (!arr.empty()) {
                if (indent >= 0) {
                    oss << '\n';
                }
                for (std::size_t i = 0; i < arr.size(); ++i) {
                    if (indent >= 0) {
                        oss << std::string((depth + 1) * indent, ' ');
                    }
                    arr[i].dumpInternal(oss, indent, depth + 1);
                    if (i + 1 < arr.size()) {
                        oss << ',';
                    }
                    if (indent >= 0) {
                        oss << '\n';
                    }
                }
                if (indent >= 0) {
                    oss << std::string(depth * indent, ' ');
                }
            }
            oss << ']';
        } else if (isObject()) {
            const auto& obj = std::get<Object>(data_);
            oss << '{';
            if (!obj.empty()) {
                if (indent >= 0) {
                    oss << '\n';
                }
                auto it = obj.begin();
                while (it != obj.end()) {
                    if (indent >= 0) {
                        oss << std::string((depth + 1) * indent, ' ');
                    }
                    oss << escapeString(it->first);
                    oss << ':';
                    if (indent >= 0) {
                        oss << ' ';
                    }
                    it->second.dumpInternal(oss, indent, depth + 1);
                    ++it;
                    if (it != obj.end()) {
                        oss << ',';
                    }
                    if (indent >= 0) {
                        oss << '\n';
                    }
                }
                if (indent >= 0) {
                    oss << std::string(depth * indent, ' ');
                }
            }
            oss << '}';
        }
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text), pos_(0) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue value = parseValue();
        skipWhitespace();
        if (pos_ != text_.size()) {
            throw std::runtime_error("Unexpected trailing characters in JSON input");
        }
        return value;
    }

private:
    const std::string& text_;
    std::size_t pos_;

    void skipWhitespace() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    char peek() const {
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of input");
        }
        return text_[pos_];
    }

    bool consume(char expected) {
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    JsonValue parseValue() {
        char ch = peek();
        if (ch == 'n') return parseNull();
        if (ch == 't' || ch == 'f') return parseBool();
        if (ch == '"') return parseString();
        if (ch == '{') return parseObject();
        if (ch == '[') return parseArray();
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) return parseNumber();
        throw std::runtime_error("Invalid JSON value");
    }

    JsonValue parseNull() {
        expect("null");
        return JsonValue(nullptr);
    }

    JsonValue parseBool() {
        if (match("true")) {
            return JsonValue(true);
        }
        if (match("false")) {
            return JsonValue(false);
        }
        throw std::runtime_error("Invalid boolean literal");
    }

    JsonValue parseNumber() {
        std::size_t start = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Invalid number");
        }
        if (text_[pos_] == '0') {
            ++pos_;
        } else if (std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        } else {
            throw std::runtime_error("Invalid number");
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                throw std::runtime_error("Invalid number");
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                throw std::runtime_error("Invalid number");
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        double value = std::stod(text_.substr(start, pos_ - start));
        return JsonValue(value);
    }

    JsonValue parseString() {
        if (!consume('"')) {
            throw std::runtime_error("Expected string");
        }
        std::string result;
        while (pos_ < text_.size()) {
            char ch = text_[pos_++];
            if (ch == '\\') {
                if (pos_ >= text_.size()) {
                    throw std::runtime_error("Invalid escape sequence");
                }
                char esc = text_[pos_++];
                switch (esc) {
                    case '\\': result.push_back('\\'); break;
                    case '"': result.push_back('"'); break;
                    case '/': result.push_back('/'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) {
                            throw std::runtime_error("Invalid unicode escape");
                        }
                        unsigned int code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char hex = text_[pos_++];
                            code <<= 4;
                            if (hex >= '0' && hex <= '9') code |= (hex - '0');
                            else if (hex >= 'a' && hex <= 'f') code |= (hex - 'a' + 10);
                            else if (hex >= 'A' && hex <= 'F') code |= (hex - 'A' + 10);
                            else throw std::runtime_error("Invalid unicode escape");
                        }
                        if (code <= 0x7F) {
                            result.push_back(static_cast<char>(code));
                        } else if (code <= 0x7FF) {
                            result.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                            result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            result.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                            result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("Invalid escape sequence");
                }
            } else if (ch == '"') {
                return JsonValue(std::move(result));
            } else {
                result.push_back(ch);
            }
        }
        throw std::runtime_error("Unterminated string");
    }

    JsonValue parseArray() {
        if (!consume('[')) {
            throw std::runtime_error("Expected array");
        }
        JsonValue::Array arr;
        skipWhitespace();
        if (consume(']')) {
            return JsonValue(std::move(arr));
        }
        while (true) {
            skipWhitespace();
            arr.push_back(parseValue());
            skipWhitespace();
            if (consume(']')) {
                break;
            }
            if (!consume(',')) {
                throw std::runtime_error("Expected ',' in array");
            }
        }
        return JsonValue(std::move(arr));
    }

    JsonValue parseObject() {
        if (!consume('{')) {
            throw std::runtime_error("Expected object");
        }
        JsonValue::Object obj;
        skipWhitespace();
        if (consume('}')) {
            return JsonValue(std::move(obj));
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                throw std::runtime_error("Expected string key in object");
            }
            std::string key = parseString().asString();
            skipWhitespace();
            if (!consume(':')) {
                throw std::runtime_error("Expected ':' after key in object");
            }
            skipWhitespace();
            obj.emplace(std::move(key), parseValue());
            skipWhitespace();
            if (consume('}')) {
                break;
            }
            if (!consume(',')) {
                throw std::runtime_error("Expected ',' in object");
            }
        }
        return JsonValue(std::move(obj));
    }

    void expect(const char* literal) {
        if (!match(literal)) {
            throw std::runtime_error(std::string("Expected '") + literal + "'");
        }
    }

    bool match(const char* literal) {
        std::size_t len = std::strlen(literal);
        if (text_.substr(pos_, len) == literal) {
            pos_ += len;
            return true;
        }
        return false;
    }
};

inline JsonValue parse(const std::string& text) {
    Parser parser(text);
    return parser.parse();
}

inline JsonValue parseFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return parse(buffer.str());
}

inline JsonValue makeObject() {
    return JsonValue(JsonValue::Object{});
}

inline JsonValue makeArray() {
    return JsonValue(JsonValue::Array{});
}

}  // namespace simplejson

