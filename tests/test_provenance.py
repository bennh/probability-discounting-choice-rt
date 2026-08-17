import csv
import tempfile
import unittest
from pathlib import Path
import zipfile

from pd_project.config import sha256_file
from pd_project.provenance import (
    MANIFEST_COLUMNS,
    append_manifest,
    materialize_archive_generation,
    raw_source_snapshot,
    sha256_named_files,
)


class RawSourceProvenanceTests(unittest.TestCase):
    def data_config(self) -> dict:
        return {
            "raw_directory": "data/raw",
            "raw_archive": "data/raw/PD data.zip",
            "raw_glob": "**/*.mat",
        }

    def test_direct_source_excludes_old_extractions_and_rejects_new_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            direct = raw / "participant.mat"
            direct.write_bytes(b"direct")
            stale = raw / "extracted" / "old" / "participant.mat"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")

            files, archive_hash = raw_source_snapshot(
                root, self.data_config(), source_mode="direct_mat"
            )
            self.assertEqual(files, [direct.resolve()])
            self.assertIsNone(archive_hash)

            (raw / "PD data.zip").write_bytes(b"new archive")
            with self.assertRaisesRegex(RuntimeError, "raw archive now exists"):
                raw_source_snapshot(root, self.data_config(), source_mode="direct_mat")

    def test_archive_source_is_bound_to_hash_and_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            archive = raw / "PD data.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("participant.mat", b"mat")
            archive_hash = sha256_file(archive)
            extracted = raw / "extracted" / archive_hash
            materialize_archive_generation(
                archive,
                extracted,
                archive_hash=archive_hash,
                raw_glob="**/*.mat",
            )
            mat = extracted / "participant.mat"

            files, observed_hash = raw_source_snapshot(
                root,
                self.data_config(),
                source_mode="archive",
                expected_archive_sha256=archive_hash,
            )
            self.assertEqual(files, [mat.resolve()])
            self.assertEqual(observed_hash, archive_hash)

            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("participant.mat", b"generation two")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                raw_source_snapshot(
                    root,
                    self.data_config(),
                    source_mode="archive",
                    expected_archive_sha256=archive_hash,
                )

    def test_archive_generation_rejects_cached_mat_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("nested/participant.mat", b"original")
            archive_hash = sha256_file(archive)
            extracted = root / "extracted" / archive_hash
            materialize_archive_generation(
                archive, extracted, archive_hash=archive_hash, raw_glob="**/*.mat"
            )
            (extracted / "nested" / "participant.mat").write_bytes(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "differs"):
                materialize_archive_generation(
                    archive,
                    extracted,
                    archive_hash=archive_hash,
                    raw_glob="**/*.mat",
                )

    def test_named_file_hash_includes_unambiguous_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left" / "participant.mat"
            right = root / "right" / "participant.mat"
            left.parent.mkdir()
            right.parent.mkdir()
            left.write_bytes(b"same")
            right.write_bytes(b"same")
            first = sha256_named_files([left, right])
            renamed = root / "right" / "renamed.mat"
            right.rename(renamed)
            second = sha256_named_files([left, renamed])
            self.assertNotEqual(first, second)

    def test_manifest_schema_is_migrated_before_append(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "artifact,stage,path\nold,prepare_data,old.csv\n",
                encoding="utf-8",
            )

            append_manifest(
                manifest,
                [
                    {
                        "artifact": "new",
                        "stage": "fit_run_a",
                        "data_pipeline_sha256": "pipeline-hash",
                        "path": "new.csv",
                    }
                ],
            )

            with manifest.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames or ()), MANIFEST_COLUMNS)
            self.assertEqual([row["artifact"] for row in rows], ["old", "new"])
            self.assertEqual(rows[0]["path"], "old.csv")
            self.assertEqual(rows[1]["data_pipeline_sha256"], "pipeline-hash")

    def test_manifest_rejects_rows_wider_than_legacy_header(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text("artifact,stage\nold,prepare_data,extra\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed manifest"):
                append_manifest(manifest, [{"artifact": "new", "stage": "fit_run_a"}])


if __name__ == "__main__":
    unittest.main()
