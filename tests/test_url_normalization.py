# tests/test_url_normalization.py

"""Unit tests for normalize_url_input() — URL normalization for CDP navigation."""

import pytest

from infrastructure.browser_use_controller import normalize_url_input


class TestSchemePassthrough:
    """URLs with scheme pass through unchanged."""

    def test_https_url(self):
        assert normalize_url_input("https://amazon.com") == "https://amazon.com"

    def test_http_url(self):
        assert normalize_url_input("http://example.org") == "http://example.org"

    def test_https_with_path(self):
        assert normalize_url_input("https://amazon.com/laptops") == "https://amazon.com/laptops"

    def test_file_url(self):
        assert normalize_url_input("file:///tmp/test.html") == "file:///tmp/test.html"

    def test_chrome_url(self):
        assert normalize_url_input("chrome://settings") == "chrome://settings"

    def test_https_with_query_string(self):
        assert normalize_url_input("https://google.com/search?q=test") == "https://google.com/search?q=test"


class TestKnownSiteAliases:
    """Known site names map to their canonical URLs."""

    def test_amazon(self):
        assert normalize_url_input("amazon") == "https://www.amazon.com"

    def test_youtube(self):
        assert normalize_url_input("youtube") == "https://www.youtube.com"

    def test_google(self):
        assert normalize_url_input("google") == "https://www.google.com"

    def test_github(self):
        assert normalize_url_input("github") == "https://github.com"

    def test_gmail(self):
        assert normalize_url_input("gmail") == "https://mail.google.com"

    def test_spotify(self):
        assert normalize_url_input("spotify") == "https://open.spotify.com"

    def test_case_insensitive(self):
        assert normalize_url_input("Amazon") == "https://www.amazon.com"
        assert normalize_url_input("YOUTUBE") == "https://www.youtube.com"

    def test_stack_overflow(self):
        assert normalize_url_input("stack overflow") == "https://stackoverflow.com"

    def test_netflix(self):
        assert normalize_url_input("netflix") == "https://www.netflix.com"

    def test_whatsapp(self):
        assert normalize_url_input("whatsapp") == "https://web.whatsapp.com"


class TestLocalhost:
    """Localhost uses http, not https."""

    def test_bare_localhost(self):
        assert normalize_url_input("localhost") == "http://localhost"

    def test_localhost_with_port(self):
        assert normalize_url_input("localhost:3000") == "http://localhost:3000"

    def test_localhost_with_port_and_path(self):
        assert normalize_url_input("localhost:8080/api") == "http://localhost:8080/api"


class TestBareDomains:
    """Bare domains (with TLD, no spaces) get https:// prepended."""

    def test_amazon_com(self):
        assert normalize_url_input("amazon.com") == "https://amazon.com"

    def test_reddit_com(self):
        assert normalize_url_input("reddit.com") == "https://reddit.com"

    def test_docs_python_org(self):
        assert normalize_url_input("docs.python.org") == "https://docs.python.org"

    def test_subdomain(self):
        assert normalize_url_input("www.amazon.com") == "https://www.amazon.com"

    def test_domain_with_path(self):
        assert normalize_url_input("amazon.com/laptops") == "https://amazon.com/laptops"

    def test_domain_with_port(self):
        assert normalize_url_input("example.com:8080") == "https://example.com:8080"

    def test_international_tld(self):
        assert normalize_url_input("example.de") == "https://example.de"

    def test_long_tld(self):
        assert normalize_url_input("example.museum") == "https://example.museum"


class TestSearchFallback:
    """Non-domain text falls back to Google search."""

    def test_single_unknown_word(self):
        result = normalize_url_input("xyzzy")
        assert result == "https://www.google.com/search?q=xyzzy"

    def test_multi_word_query(self):
        result = normalize_url_input("best laptops")
        assert result.startswith("https://www.google.com/search?q=")
        assert "best" in result
        assert "laptops" in result

    def test_python_tutorial(self):
        result = normalize_url_input("python tutorial")
        assert result.startswith("https://www.google.com/search?q=")

    def test_special_characters_encoded(self):
        result = normalize_url_input("what is AI?")
        assert result.startswith("https://www.google.com/search?q=")
        # Question mark should be URL-encoded
        assert "?" in result.split("q=")[0]  # The ? in query string
        assert "what" in result


class TestEdgeCases:
    """Edge cases: whitespace, empty, trailing slashes."""

    def test_empty_string(self):
        assert normalize_url_input("") == ""

    def test_whitespace_only(self):
        assert normalize_url_input("   ") == ""

    def test_leading_trailing_whitespace(self):
        assert normalize_url_input("  amazon  ") == "https://www.amazon.com"

    def test_leading_whitespace_url(self):
        assert normalize_url_input("  https://google.com  ") == "https://google.com"
