"""CLI argument handling and exit codes.

None of these touch the hub: ``info`` only inspects the registry, and the
failure cases are rejected before any model is looked at.
"""

import json

import pytest

from airavata_quant.cli import _build_parser, _positive_int, main


def test_info_prints_json_for_every_variant(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["--device", "cpu", "info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu"
    assert set(payload["variants"]) == {"original", "int8", "int4", "dynamic_quant"}
    assert payload["variants"]["int4"]["supported_here"] is False
    # A configured token must never be echoed back, only its presence.
    assert payload["hf_token_configured"] in (True, False)
    assert "hf_token" not in payload


def test_a_bad_log_level_is_a_configuration_error_not_a_traceback(capsys, monkeypatch, tmp_path):
    """It used to reach uvicorn and blow up there."""
    monkeypatch.chdir(tmp_path)
    assert main(["--log-level", "verbose", "info"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_a_bad_port_override_is_rejected_before_binding(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["serve", "--port", "99999"]) == 2
    assert "port" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["benchmark", "--iterations", "0"],
                                  ["benchmark", "--max-new-tokens", "-5"],
                                  ["serve", "--port", "abc"]])
def test_non_positive_integer_arguments_are_rejected_by_the_parser(argv):
    with pytest.raises(SystemExit) as excinfo:
        _build_parser().parse_args(argv)
    assert excinfo.value.code == 2


def test_positive_int_accepts_valid_values():
    assert _positive_int("7") == 7


def test_a_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])
