from __future__ import annotations

from podaddeduct import db
from podaddeduct.config import settings
from podaddeduct.feeds import generate_custom_feed_xml, parse_feed, rewrite_feed_xml
from podaddeduct.intervals import Interval, invert_ranges, merge_intervals


def test_merge_and_invert():
    ads = merge_intervals([Interval(0, 10), Interval(8, 15), Interval(100, 120)])
    assert [(a.start, a.end) for a in ads] == [(0, 15), (100, 120)]
    content = invert_ranges(ads, 130.0)
    assert [(c.start, c.end) for c in content] == [(15, 100), (120, 130)]


def test_feed_rewrite_points_at_clean_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir()
    db.init_db()
    feed = db.create_feed(slug="test-feed", upstream_url="https://example.com/rss", title="Test")
    enc = "https://example.com/ep1.mp3"
    ep = db.upsert_episode(feed.id, guid="g1", title="E1", enclosure_url=enc, pub_date=None)
    # Simulate a cut MP3
    clean = tmp_path / "audio" / f"{ep.id}.clean.mp3"
    clean.write_bytes(b"x" * 100)
    db.update_episode(ep.id, audio_path=str(tmp_path / "audio" / f"{ep.id}.bin"),
                      duration_seconds=300.0, clean_audio_path=str(clean),
                      ad_ranges_json='[{"start": 10, "end": 20}]', status="ready")
    ep = db.get_episode(ep.id)
    assert ep
    rss = (
        b'<?xml version="1.0"?><rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">'
        b"<channel><title>T</title><link>http://x</link><description>d</description>"
        b'<item><title>E1</title><guid isPermaLink="false">g1</guid>'
        b"<pubDate>Mon, 01 Sep 2025 00:00:00 GMT</pubDate>"
        b'<enclosure url="https://example.com/ep1.mp3" length="999" type="audio/mpeg"/></item>'
        b"</channel></rss>"
    )
    parsed = parse_feed(rss)
    xml = rewrite_feed_xml(parsed, feed=feed, episodes_by_guid={"g1": ep},
                           public_base="http://192.168.0.93:8080")
    assert f"/audio/{ep.id}" in xml
    assert f"/audio/{ep.id}?v=" in xml
    assert 'length="100"' in xml  # clean file size, not upstream 999
    assert "<itunes:duration>300</itunes:duration>" in xml
    assert "podcast:chapters" not in xml


def test_generate_custom_feed_includes_pending(tmp_path, monkeypatch):
    """Pending episodes stay in the player feed; /audio 302s until cleaned."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir()
    db.init_db()
    feed = db.create_feed(slug="show", upstream_url="https://example.com/rss", title="My Show")
    db.update_feed_channel(feed.id, description="Show notes here", author="Host")
    db.update_feed_artwork(feed.id, "https://cdn.example.com/art.jpg")
    feed = db.get_feed(feed.id)
    assert feed

    pending = db.upsert_episode(
        feed.id, guid="g-pending", title="Pending Ep",
        enclosure_url="https://example.com/p.mp3",
        pub_date="Mon, 01 Sep 2025 00:00:00 GMT",
    )
    ready = db.upsert_episode(
        feed.id, guid="g-ready", title="Ready Ep",
        enclosure_url="https://example.com/r.mp3",
        pub_date="Tue, 02 Sep 2025 00:00:00 GMT",
    )
    clean = tmp_path / "audio" / f"{ready.id}.clean.mp3"
    clean.write_bytes(b"y" * 50)
    db.update_episode(
        ready.id,
        status="ready",
        clean_audio_path=str(clean),
        duration_seconds=120.0,
        description="Episode show notes",
        size_bytes=50,
    )
    ready = db.get_episode(ready.id)
    assert ready

    # Ready-only helper still filters; the player feed lists every tracked row.
    ready_eps = db.list_ready_episodes(feed.id)
    assert [e.guid for e in ready_eps] == ["g-ready"]
    all_eps = db.list_episodes(feed.id)
    assert {e.guid for e in all_eps} == {"g-ready", "g-pending"}

    xml = generate_custom_feed_xml(feed, all_eps, "http://192.168.0.93:8080")
    assert "<title>My Show</title>" in xml
    assert "Show notes here" in xml
    assert 'href="https://cdn.example.com/art.jpg"' in xml
    assert "Ready Ep" in xml
    assert "Pending Ep" in xml
    assert f"/audio/{ready.id}" in xml
    assert f"/audio/{ready.id}?v=" in xml
    assert f"/audio/{pending.id}" in xml
    assert 'length="50"' in xml
    assert "<itunes:duration>120</itunes:duration>" in xml
    assert "Episode show notes" in xml
    assert xml.count("<item>") == 2


def test_generate_custom_feed_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    (tmp_path / "audio").mkdir()
    db.init_db()
    feed = db.create_feed(slug="empty", upstream_url="https://example.com/e", title="Empty")
    xml = generate_custom_feed_xml(feed, [], "http://localhost:8080")
    assert "<title>Empty</title>" in xml
    assert "<item>" not in xml
