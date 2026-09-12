from __future__ import annotations

from unittest.mock import patch

from podaddeduct.chapters import (
    ad_ranges_from_chapters_json,
    is_ad_chapter_title,
    parse_chapters_tags,
    try_publisher_chapters,
)
from podaddeduct.db import Episode


def test_is_ad_chapter_title():
    assert is_ad_chapter_title("Ad")
    assert is_ad_chapter_title("Sponsor")
    assert is_ad_chapter_title("Ad: Acme Insurance")
    assert is_ad_chapter_title("Mid-roll")
    assert not is_ad_chapter_title("Content")
    assert not is_ad_chapter_title("Interview with the coach")
    assert not is_ad_chapter_title("")


def test_ad_ranges_from_chapters_json():
    data = {
        "version": "1.2.0",
        "chapters": [
            {"startTime": 0, "title": "Intro", "endTime": 30},
            {"startTime": 30, "title": "Ad", "endTime": 90},
            {"startTime": 90, "title": "Main", "endTime": 500},
            {"startTime": 500, "title": "Sponsor: Foo", "endTime": 560},
            {"startTime": 560, "title": "Outro"},
        ],
    }
    ads = ad_ranges_from_chapters_json(data, duration=600)
    assert len(ads) == 2
    assert ads[0].start == 30 and ads[0].end == 90
    assert ads[1].start == 500 and ads[1].end == 560


def test_ad_ranges_infers_end_from_next():
    data = {
        "chapters": [
            {"startTime": 10, "title": "Ad"},
            {"startTime": 40, "title": "Content"},
        ]
    }
    ads = ad_ranges_from_chapters_json(data)
    assert len(ads) == 1
    assert ads[0].start == 10 and ads[0].end == 40


def test_parse_chapters_tags():
    raw = b"""<?xml version="1.0"?>
    <rss xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel>
      <item>
        <guid>ep-1</guid>
        <enclosure url="https://cdn.example/a.mp3" />
        <podcast:chapters url="https://cdn.example/a.json" type="application/json+chapters" />
      </item>
      <item>
        <guid>ep-2</guid>
        <enclosure url="https://cdn.example/b.mp3" />
      </item>
      <item>
        <guid>ep-3</guid>
        <enclosure url="https://cdn.example/c.mp3" />
        <podcast:chapters url="https://cdn.example/c.psc" type="application/x-psc" />
      </item>
    </channel></rss>"""
    m = parse_chapters_tags(raw)
    assert m["ep-1"] == "https://cdn.example/a.json"
    assert m["https://cdn.example/a.mp3"] == "https://cdn.example/a.json"
    assert "ep-2" not in m
    assert "ep-3" not in m  # non-JSON chapter type ignored (no NameError)


def test_try_publisher_chapters_returns_ads():
    ep = Episode(
        id=1, feed_id=1, guid="g", title="T", enclosure_url="https://e/a.mp3",
        pub_date=None, chapters_url="https://cdn.example/ch.json",
    )
    payload = {
        "chapters": [
            {"startTime": 0, "title": "Show", "endTime": 100},
            {"startTime": 100, "title": "Ad", "endTime": 160},
        ]
    }

    class _Resp:
        content = b"x" * 10
        def raise_for_status(self):
            return None
        def json(self):
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, url):
            assert url == "https://cdn.example/ch.json"
            return _Resp()

    with patch("podaddeduct.chapters.httpx.Client", _Client):
        ads = try_publisher_chapters(ep, duration=200)
    assert ads is not None
    assert len(ads) == 1
    assert ads[0].start == 100


def test_try_publisher_chapters_none_without_url():
    ep = Episode(
        id=1, feed_id=1, guid="g", title="T", enclosure_url="https://e/a.mp3",
        pub_date=None, chapters_url=None,
    )
    assert try_publisher_chapters(ep) is None
