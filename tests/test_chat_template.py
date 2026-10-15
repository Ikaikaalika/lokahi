"""Correctness evidence: the native chat-template renderer against jinja2.

transformers' ``apply_chat_template`` renders a template with jinja2's
``ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`` plus the
loopcontrols extension, its own ``tojson``, ``raise_exception`` and
``strftime_now``, and a ``{% generation %}`` tag, passing ``messages``,
``tools=None``, ``documents=None``, ``add_generation_prompt`` and the special
tokens. The oracle below replicates that setup and every comparison is byte for
byte. The real templates are pinned in ``tests/fixtures/chat_templates`` (see
its README); the synthetic templates exercise each construct on its own.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")
from jinja2 import nodes  # noqa: E402
from jinja2.ext import Extension, loopcontrols  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "chat_templates"
MODELS = sorted(path.name for path in FIXTURES.iterdir() if path.is_dir())
# Templates that print today's date unless given date_string.
DATED = {"gpt-oss", "llama3.2", "smollm3"}


# --- oracle ------------------------------------------------------------------


class _Generation(Extension):
    """transformers' AssistantTracker without the token-mask bookkeeping."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

    def _render(self, caller):
        return caller()


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


ENV = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[_Generation, loopcontrols])
ENV.filters["tojson"] = _tojson
ENV.globals["raise_exception"] = _raise_exception
ENV.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)


def oracle(source, messages, *, add_generation_prompt=True, tokens=None, variables=None):
    """How apply_chat_template ends: ("ok", text), ("raise", message) or ("error", type)."""
    try:
        template = ENV.from_string(source)
    except jinja2.TemplateSyntaxError as error:
        return ("error", type(error).__name__)
    kwargs = {key: value for key, value in (tokens or {}).items() if value is not None}
    kwargs.update(variables or {})
    # A copy: jinja2's |indent extends a list argument in place before it fails.
    messages = copy.deepcopy(messages)
    try:
        return ("ok", template.render(messages=messages, tools=None, documents=None,
                                      add_generation_prompt=add_generation_prompt, **kwargs))
    except jinja2.TemplateError as error:
        if type(error) is jinja2.TemplateError:  # raise_exception
            return ("raise", error.message)
        return ("error", type(error).__name__)
    except Exception as error:  # noqa: BLE001 - any Python error ends apply_chat_template
        return ("error", type(error).__name__)


def _token(value):
    return value.get("content") if isinstance(value, dict) else value


def snapshot(model):
    """The template and special tokens transformers loads from a snapshot directory."""
    directory = FIXTURES / model
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    jinja = directory / "chat_template.jinja"
    source = jinja.read_text(encoding="utf-8") if jinja.exists() else config["chat_template"]
    return source, {"bos_token": _token(config.get("bos_token")), "eos_token": _token(config.get("eos_token"))}


# --- native ------------------------------------------------------------------

_counter = itertools.count()


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("chat-template")


def _write(workdir, suffix, text):
    path = workdir / f"{next(_counter)}{suffix}"
    path.write_text(text, encoding="utf-8")
    return path


def native(cli, workdir, messages, *, model=None, source=None, tokens=None, add_generation_prompt=True,
           variables=None, extra=()):
    """("ok", text) or ("error", stderr) from `lokahi chat-template`."""
    args = [str(cli), "chat-template",
            "--messages-file", str(_write(workdir, ".json", json.dumps(messages, ensure_ascii=False)))]
    if model is not None:
        args += ["--model", str(FIXTURES / model)]
    else:
        args += ["--template-file", str(_write(workdir, ".jinja", source))]
        if tokens:
            config = {key: value for key, value in tokens.items() if value is not None}
            args += ["--config-file", str(_write(workdir, ".json", json.dumps(config)))]
    if not add_generation_prompt:
        args.append("--no-generation-prompt")
    if variables:
        args += ["--vars-file", str(_write(workdir, ".json", json.dumps(variables, ensure_ascii=False)))]
    args += list(extra)
    result = subprocess.run(args, capture_output=True)
    if result.returncode == 0:
        return ("ok", result.stdout.decode("utf-8"))
    assert result.returncode == 2, result.stderr
    return ("error", result.stderr.decode("utf-8").rstrip("\n"))


def assert_same(actual, expected):
    """Native output equals the oracle's; raise_exception messages are kept verbatim."""
    if expected[0] == "ok":
        assert actual == expected
    elif expected[0] == "raise":
        assert actual == ("error", expected[1])
    else:
        assert actual[0] == "error", f"jinja2 raised {expected[1]} but native rendered {actual[1]!r}"


def check_model(cli, workdir, model, messages, *, add_generation_prompt=True, variables=None):
    """Compares a fixture snapshot; dated templates are checked against the oracle
    just before and after the native call so midnight cannot flake."""
    source, tokens = snapshot(model)

    def expected():
        return oracle(source, messages, add_generation_prompt=add_generation_prompt, tokens=tokens,
                      variables=variables)

    before = expected()
    actual = native(cli, workdir, messages, model=model, add_generation_prompt=add_generation_prompt,
                    variables=variables)
    after = expected()
    assert_same(actual, after if actual == after else before)
    return before


# --- fixtures ------------------------------------------------------------------


def test_fixture_readme_records_every_file():
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    files = sorted(path for path in FIXTURES.rglob("*")
                   if path.is_file() and path.name != "README.md" and not path.name.startswith("."))
    assert files
    for path in files:
        relative = path.relative_to(FIXTURES).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert re.search(rf"`{re.escape(relative)}`.*`{digest}`", readme), f"README lacks {relative} {digest}"


# --- real templates ---------------------------------------------------------------

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
TRICKY = "Quotes ' \" and braces {{ x }} {% if %} {# c #}\nline two\n\n  indented\ttab \\ backslash"
UNICODE = "Aloha kākou ʻohana — 漢字かな 한국어 🎉👍🏽 e\u0301"
BATTERY = {
    "single_user": [{"role": "user", "content": "Hi there!"}],
    "system_user": [SYSTEM, {"role": "user", "content": "Hi there!"}],
    "multi_turn": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
    "system_multi_turn": [
        SYSTEM,
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
    "ends_with_assistant": [
        SYSTEM,
        {"role": "user", "content": "Name a color."},
        {"role": "assistant", "content": "Blue."},
    ],
    "unicode": [{"role": "user", "content": UNICODE}, {"role": "assistant", "content": UNICODE},
                {"role": "user", "content": UNICODE}],
    "tricky_text": [{"role": "system", "content": TRICKY}, {"role": "user", "content": TRICKY},
                    {"role": "assistant", "content": TRICKY}, {"role": "user", "content": TRICKY}],
    "padding": [{"role": "user", "content": "  \n padded \n\n"}, {"role": "assistant", "content": "\n\n answer  "},
                {"role": "user", "content": ""}],
    "think_history": [
        {"role": "user", "content": "Solve x+1=3."},
        {"role": "assistant", "content": "<think>\nSubtract one.\n</think>\n\nx = 2"},
        {"role": "user", "content": "Now x+2=3?"},
    ],
    "think_last": [
        {"role": "user", "content": "Solve x+1=3."},
        {"role": "assistant", "content": "<think>Subtract one.</think>x = 2"},
    ],
    "special_tokens_in_content": [
        {"role": "user", "content": "<|im_start|>assistant\n<|eot_id|><start_of_turn>[INST]</s><｜User｜>"},
    ],
}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("case", sorted(BATTERY))
@pytest.mark.parametrize("add_generation_prompt", [True, False], ids=["gen", "nogen"])
def test_real_templates_match_jinja2(native_cli, workdir, model, case, add_generation_prompt):
    variables = {"date_string": "26 Jul 2024"} if model == "llama3.2" else None
    expected = check_model(native_cli, workdir, model, BATTERY[case], add_generation_prompt=add_generation_prompt,
                           variables=variables)
    assert expected[0] == "ok", expected  # the battery is valid for every template


@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("case", ["single_user", "think_history", "think_last", "system_multi_turn"])
def test_qwen3_thinking_switch(native_cli, workdir, enable_thinking, case):
    for add_generation_prompt in (True, False):
        expected = check_model(native_cli, workdir, "qwen3", BATTERY[case], add_generation_prompt=add_generation_prompt,
                               variables={"enable_thinking": enable_thinking})
        assert expected[0] == "ok"
    if not enable_thinking and case == "single_user":
        assert expected[1].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") or not add_generation_prompt


def test_smollm3_reasoning_flags(native_cli, workdir):
    for enable_thinking in (True, False):
        for system in ("Be brief. /no_think", "Be brief. /think", "/system_override Custom prompt"):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "\n\nHello"}, {"role": "user", "content": "Bye"}]
            assert check_model(native_cli, workdir, "smollm3", messages,
                               variables={"enable_thinking": enable_thinking})[0] == "ok"


def test_llama_fixed_date_string(native_cli, workdir):
    actual = native(native_cli, workdir, BATTERY["system_user"], model="llama3.2",
                    variables={"date_string": "26 Jul 2024"})
    assert actual[0] == "ok"
    assert "Today Date: 26 Jul 2024\n\n" in actual[1]
    source, tokens = snapshot("llama3.2")
    assert actual == oracle(source, BATTERY["system_user"], tokens=tokens, variables={"date_string": "26 Jul 2024"})


def test_strftime_now_renders_today_in_llama_format(native_cli, workdir):
    before = datetime.now().strftime("%d %b %Y")
    actual = native(native_cli, workdir, BATTERY["single_user"], model="llama3.2")
    after = datetime.now().strftime("%d %b %Y")
    assert actual[0] == "ok"
    assert f"Today Date: {before}\n\n" in actual[1] or f"Today Date: {after}\n\n" in actual[1]
    check_model(native_cli, workdir, "llama3.2", BATTERY["single_user"])


def test_cli_set_and_flag_options(native_cli, workdir):
    source, tokens = snapshot("llama3.2")
    actual = native(native_cli, workdir, BATTERY["single_user"], model="llama3.2",
                    extra=["--set", "date_string=01 Jan 2025"])
    assert actual == oracle(source, BATTERY["single_user"], tokens=tokens, variables={"date_string": "01 Jan 2025"})
    source, tokens = snapshot("qwen3")
    actual = native(native_cli, workdir, BATTERY["single_user"], model="qwen3",
                    extra=["--flag", "enable_thinking=false,unused=true"])
    assert actual == oracle(source, BATTERY["single_user"], tokens=tokens, variables={"enable_thinking": False})


def test_template_file_with_config_file(native_cli, workdir):
    source, tokens = snapshot("gemma3")
    actual = native(native_cli, workdir, BATTERY["system_multi_turn"], source=source, tokens=tokens)
    assert actual == oracle(source, BATTERY["system_multi_turn"], tokens=tokens)
    assert actual[1].startswith("<bos><start_of_turn>user\n")


def test_jinja_file_takes_precedence_over_config_template(native_cli, workdir, tmp_path):
    (tmp_path / "chat_template.jinja").write_text("jinja {{ eos_token }}\n", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "config", "eos_token": "</s>", "bos_token": None}), encoding="utf-8")
    assert native(native_cli, workdir, BATTERY["single_user"], model=tmp_path) == ("ok", "jinja </s>")
    # GLM-4 ships both; its inline template differs only in the metadata branch.
    config = json.loads((FIXTURES / "glm4-0414" / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["chat_template"] != (FIXTURES / "glm4-0414" / "chat_template.jinja").read_text(encoding="utf-8")


def test_long_conversation(native_cli, workdir):
    messages = [{"role": "system", "content": "s" * 2000}]
    for i in range(150):
        messages.append({"role": "user", "content": f"question {i} " + "ʻ漢🎉" * 50})
        messages.append({"role": "assistant", "content": f"<think>r{i}</think>answer {i} " + "x" * 300})
    messages.append({"role": "user", "content": "last"})
    for model in MODELS:
        variables = {"date_string": "26 Jul 2024"} if model == "llama3.2" else None
        assert check_model(native_cli, workdir, model, messages, variables=variables)[0] == "ok"


# --- errors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,messages",
    [
        ("gemma3", [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]),
        ("gemma3", [{"role": "assistant", "content": "a"}]),
        ("mistral-v0.3", [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]),
        ("mistral-v0.3", [{"role": "developer", "content": "a"}]),
        ("gpt-oss", [{"role": "user", "content": "a"},
                     {"role": "assistant", "content": "<|channel|>analysis<|message|>leak"}]),
        ("gpt-oss", [{"role": "tool", "content": "orphan"}]),
    ],
)
def test_raise_exception_message_is_preserved(native_cli, workdir, model, messages):
    source, tokens = snapshot(model)
    expected = oracle(source, messages, tokens=tokens)
    assert expected[0] == "raise"
    assert native(native_cli, workdir, messages, model=model) == ("error", expected[1])


UNSUPPORTED = {
    "include": "{% include 'other.jinja' %}",
    "extends": "{% extends 'base.jinja' %}",
    "raw": "a\n{% raw %}{{ x }}{% endraw %}",
    "filter_block": "{% filter upper %}x{% endfilter %}",
    "with": "{% with a = 1 %}{{ a }}{% endwith %}",
    "recursive_loop": "{% for m in messages recursive %}{{ loop(m) }}{% endfor %}",
    "unknown_filter": "{{ messages|wordcount }}",
    "unknown_test": "{% if messages is callable %}x{% endif %}",
    "unknown_method": "{{ 'x'.zfill(3) }}",
    "loop_cycle": "{% for m in messages %}{{ loop.cycle('a', 'b') }}{% endfor %}",
    "cycler": "{% set c = cycler('a', 'b') %}",
    "template_reference": "{{ self }}",
    "unknown_function": "{{ lookup('x') }}",
    "star_args": "{{ range(*[1, 2]) }}",
    "filtered_set_block": "{% set x | upper %}a{% endset %}",
    "unknown_tag": "{% frobnicate %}",
    "unclosed_if": "{% if true %}x",
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unsupported_syntax_fails_at_parse(native_cli, workdir, name):
    source = UNSUPPORTED[name]
    path = _write(workdir, ".jinja", source)
    # The messages file does not exist: an error about the template proves it was rejected first.
    result = subprocess.run([str(native_cli), "chat-template", "--template-file", str(path),
                             "--messages-file", str(workdir / "missing.json")], capture_output=True, text=True)
    assert result.returncode == 2
    assert re.match(r"lokahi: chat template line \d+: ", result.stderr), result.stderr


def test_snapshot_without_template_fails(native_cli, workdir, tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}), encoding="utf-8")
    actual = native(native_cli, workdir, BATTERY["single_user"], model=tmp_path)
    assert actual[0] == "error" and "no chat template" in actual[1]
    empty = tmp_path / "empty"
    empty.mkdir()
    actual = native(native_cli, workdir, BATTERY["single_user"], model=empty)
    assert actual[0] == "error" and "no chat template" in actual[1]


def test_named_template_list(native_cli, workdir, tmp_path):
    templates = [{"name": "tool_use", "template": "tools"}, {"name": "default", "template": "{{ bos_token }}default"}]
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": templates, "bos_token": {"content": "<s>", "lstrip": False}}), encoding="utf-8")
    assert native(native_cli, workdir, BATTERY["single_user"], model=tmp_path) == ("ok", "<s>default")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": templates[:1]}), encoding="utf-8")
    actual = native(native_cli, workdir, BATTERY["single_user"], model=tmp_path)
    assert actual[0] == "error" and "default" in actual[1]


# --- synthetic templates --------------------------------------------------------------

SYNTHETIC = [
    # Whitespace control, trim_blocks and lstrip_blocks.
    "a\n  {% if true %}\n  b\n  {% endif %}\nc",
    "a {%- if true -%} b {%+ if true %} c {% endif %}{% endif %}\n d",
    "  {{ 'x' }}\n  {# c #}\n  y {#- z -#}  w",
    "{%- if true %}\n  x\n{%- endif %}",
    "\n  {% if true -%}\n   x\n  {%- endif %}\n",
    "a\n\n{% if true %}\n\nb{% endif %}",
    "x\n  {%+ if true %}y{% endif %}",
    "x\n  {% if true +%}\ny{% endif %}",
    "{{ 'a' }}  \n  {{ 'b' }}",
    "{% if true %}\t \n{% endif %}",
    "\t{% if true %}\u3000\n\u3000{% endif %}",
    "{{- ' x ' -}}  \n {{- 'y' }}  \n",
    "line1\r\nline2\r{% if true %}\r\nz{% endif %}\r\n",
    "{# multi\nline\ncomment #}\nafter\n\n",
    "{%- for m in messages -%}\n  [{{ m.role }}]\n{%- endfor %}\n",
    "{% for m in messages %}\n{{ m.content }}\n{% endfor %}",
    "{% for m in messages %}  {% if loop.first %}\n first{% endif %}\n{% endfor %}",
    "  {#- a -#}  \n  b{# x -#}  \n{#+ y #}\n  c",
    # Undefined values.
    "{{ x }}|{{ x is defined }}|{{ x is undefined }}|{{ x|default('d') }}",
    "{{ x.y }}",
    "{{ messages[0].missing }}|{{ messages[0].missing is defined }}|{{ messages[0]['missing'] is defined }}",
    "{{ messages[0].missing.deeper }}",
    "{{ {'a': 1}['b'] is defined }} {{ [1][5] is defined }} {{ messages[9] is defined }}",
    "{{ x ~ 'a' }} {{ x|length }} {{ x|list }} {{ x|join }} {{ x|first is defined }} {{ x|trim }}|",
    "{{ x + 1 }}",
    "{{ none.x }}|{{ 'ab'.x is defined }}|{{ (1).real is defined if false else 1 }}",
    "{{ loop.index }}",
    "{{ [x] }} {{ [none, x] }}",
    # Truthiness.
    "{% for v in ['', 'a', [], [0], {}, {'a': 1}, none, 0, 1, 0.0, 0.5, true, false] %}"
    "{{ 'T' if v else 'F' }}{% endfor %}",
    "{{ [] or 'e' }} {{ 'x' and 'y' }} {{ 0 or none }} {{ not '' }} {{ x or 'z' }} {{ x and 1 }}",
    "{% if []|select %}T{% else %}F{% endif %}{% if []|list %}T{% else %}F{% endif %}",
    # String literals and slicing.
    "{{ 'a\\nb\\tc\\\\d\\'e\\\"f\\x41\\u00e9\\101\\q' }}",
    "{{ \"it's\" }} {{ 'say \"hi\"' }} {{ 'a' 'b' \"c\" }} {{ 'é\\x41' }}",
    "{{ 'abc'[::-1] }} {{ 'héllo'[1:3] }} {{ 'ʻokina'[1:] }} {{ '漢字🎉'[::-1] }} {{ 'abcdef'[::2] }}"
    " {{ 'abcdef'[-2:] }} {{ 'ab'[-1] }} {{ 'ab'[5] is defined }} {{ 'abc'[1:100] }} {{ 'abc'[-100:1] }}",
    "{{ [1, 2, 3][1:] }} {{ [1, 2, 3][:-1] }} {{ [1, 2, 3][::-1] }} {{ [1, 2, 3][::-2] }} {{ (1, 2, 3)[1:] }}"
    " {{ messages[1:]|length }} {{ messages[-1].role }} {{ [1, 2][0:1:None] }}",
    "{{ [1][0:1:0] }}",
    # Operators.
    "{{ 1 + 2 }} {{ 7 - 10 }} {{ 3 * 4 }} {{ 7 / 2 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }}"
    " {{ 2 ** 10 }} {{ 2 ** -1 }} {{ 0 ** 0 }} {{ 10 ** 18 }}",
    "{{ 1.5 + 1 }} {{ 0.1 + 0.2 }} {{ 1e16 }} {{ 1e15 }} {{ 0.0001 }} {{ 1e-5 }} {{ 1 / 3 }} {{ 10 / 2 }}"
    " {{ -0.0 }} {{ 3 // 2.0 }} {{ -3 % 2.0 }} {{ 7.5 // 2 }} {{ 2 ** 0.5 }} {{ 7 % 2.5 }} {{ 1 / 3 * 3 }}",
    "{{ 'a' ~ 1 ~ none ~ true ~ 1.0 ~ [1, 'b'] ~ x }}",
    "{{ 'ab' * 3 }} {{ [1] * 2 }} {{ [1] + [2] }} {{ (1,) + (2,) }} {{ true + true }} {{ -true }} {{ 2 * 'x' }}"
    " {{ 'ab' * -1 }}|",
    "{{ 'a' in 'abc' }} {{ 'z' not in 'abc' }} {{ 1 in [1, 2] }} {{ 'k' in {'k': 1} }} {{ 3 not in (1, 2) }}"
    " {{ 'content' in messages[0] }} {{ 'x' in u }}",
    "{{ 1 == 1.0 }} {{ (1, 2) == [1, 2] }} {{ {'a': 1, 'b': 2} == {'b': 2, 'a': 1} }} {{ 'a' < 'b' }}"
    " {{ [1, 2] < [1, 3] }} {{ 1 < 2 < 3 }} {{ 3 > 2 > 2 }} {{ none == none }} {{ u == none }} {{ u != 1 }}"
    " {{ messages[0] == messages[0] }} {{ 'b' >= 'a' }} {{ 1.5 <= 1 }}",
    "{{ 'x' if true else 'y' }} {{ 'x' if false else 'y' }} {{ 'x' if false }}|{{ ('x' if false) is defined }}"
    " {{ 'a' if '' else 'b' if 0 else 'c' }}",
    "{{ -2 ** 2 }} {{ 2 ** 3 ** 2 }} {{ -3|abs }} {{ (1 + 2) * 3 }} {{ 1 + 2 * 3 }} {{ 2 - -1 }} {{ +3 }}",
    "{{ 'a' + 1 }}",
    "{{ 1 / 0 }}",
    "{{ 1 < 'a' }}",
    "{{ 'x' in none }}",
    # Tests.
    "{{ none is none }} {{ 1 is number }} {{ true is number }} {{ 1.5 is float }} {{ 1 is integer }}"
    " {{ true is integer }} {{ 'a' is string }} {{ {} is mapping }} {{ [] is mapping }} {{ 'a' is iterable }}"
    " {{ 3 is iterable }} {{ [] is sequence }} {{ {} is sequence }} {{ x is sequence }} {{ x is iterable }}"
    " {{ true is boolean }} {{ true is true }} {{ 0 is false }} {{ false is false }} {{ none is iterable }}",
    "{{ 1 is eq 1 }} {{ 1 is equalto 2 }} {{ 1 is ne 2 }} {{ 1 is lt 2 }} {{ 2 is gt 1 }} {{ 1 is le 1 }}"
    " {{ 1 is ge 2 }} {{ 1 is in [1] }} {{ 3 is odd }} {{ 4 is even }} {{ 9 is divisibleby 3 }}"
    " {{ 10 is divisibleby(4) }} {{ x is not defined }} {{ 'a' is not string }} {{ 4.0 is even }}"
    " {{ 1 is lessthan 2 }} {{ 1 is greaterthan 2 }} {{ 'a' is in 'abc' }} {{ messages is sequence }}",
    "{{ messages[0].content is string }} {{ messages[0].tool_calls is not defined }}"
    " {{ messages[0].content.content is defined }} {{ messages|length is odd }}",
    # Loops.
    "{% for x in 'abc' %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }}"
    "{{ loop.revindex }}{{ loop.revindex0 }}[{{ loop.previtem }}|{{ loop.nextitem }}]{{ loop.depth }};{% endfor %}",
    "{% for m in messages if m.role != 'system' %}{{ loop.index }}/{{ loop.length }}:{{ m.role }} "
    "{% else %}empty{% endfor %}",
    "{% for x in [] %}x{% else %}E{% set y = 1 %}{{ loop is defined }}{% endfor %}{{ y }}",
    "{% for i in [1, 2, 3, 4, 5] %}{% if i == 2 %}{% continue %}{% endif %}{% if i == 4 %}{% break %}{% endif %}"
    "{{ i }}{% endfor %}",
    "{% for k, v in {'a': 1, 'b': 2}.items() %}{{ k }}={{ v }};{% endfor %}"
    "{% for (a, b) in [(1, 2), [3, 4]] %}{{ a }}{{ b }}{% endfor %}{% for k, v in {'z': 0}|items %}{{ k }}{% endfor %}",
    "{% for k in {'x': 1, 'y': 2} %}{{ k }}{% endfor %}{% for c in 'ʻ漢' %}<{{ c }}>{% endfor %}",
    "{% for a in [1, 2] %}{% for b in 'xy' %}{{ loop.index }}{{ a }}{{ b }} {% endfor %}{{ loop.index }}|{% endfor %}",
    "{% for i in range(3) %}{{ i }}{% endfor %}{% for i in range(2, 9, 3) %}{{ i }}{% endfor %}{{ range(4)|list }}"
    " {{ range(5)|length }} {{ range(0)|list }} {% for i in range(3, 0, -1) %}{{ i }}{% endfor %}",
    "{% for message in messages[::-1] %}{{ (messages|length - 1) - loop.index0 }}{{ message.role[0] }}{% endfor %}",
    "{% for a, b in [[1, 2, 3]] %}{% endfor %}",
    "{% for c in none %}{% endfor %}",
    # Scoping.
    "{% set x = 0 %}{% for i in [1, 2, 3] %}{{ x }}{% set x = i %}{{ x }};{% endfor %}|{{ x }}",
    "{% set x = 5 %}{% for i in [1, 2] %}{% if i == 2 %}{% set x = 9 %}{% endif %}{{ x }}{% endfor %}{{ x }}",
    "{% if true %}{% set z = 1 %}{% endif %}{{ z }}",
    "{% for i in [1] %}{% endfor %}{{ i }}|{% set i = 7 %}{% for i in [1] %}{{ i }}{% endfor %}{{ i }}",
    "{% set ns = namespace(a=1, b='x') %}{% for i in [1, 2] %}{% set ns.a = ns.a + i %}{% endfor %}"
    "{{ ns.a }}{{ ns.b }}{{ ns.missing is defined }}{{ ns }}",
    "{% set ns = namespace() %}{% set ns.found = false %}{% for m in messages %}{% if m.role == 'user' %}"
    "{% set ns.found = true %}{% endif %}{% endfor %}{{ ns.found }}",
    "{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set c, d = 'xy' %}{{ c }}{{ d }}{% set t = 1, 2 %}{{ t }}",
    "{% set x %}block {{ 1 + 1 }}{% set inner = 1 %}{% endset %}[{{ x }}]{{ x is string }}{{ inner }}",
    "{% set messages = messages[1:] %}{{ messages|length }}",
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{ add_generation_prompt }}",
    "{% set ns = 5 %}{% set ns.a = 1 %}",
    # Macros and call blocks.
    "{% macro greet(name, punct='!') %}Hi {{ name }}{{ punct }}{% endmacro %}{{ greet('a') }} {{ greet('b', '?') }}"
    " {{ greet(punct='.', name='c') }}",
    "{% macro m(a, b) %}{{ a }}|{{ b }}|{{ b is defined }}{% endmacro %}{{ m(1) }}",
    "{% macro m(a, b=a) %}{{ b }}{% endmacro %}{{ m(3) }}{% set g = 5 %}{% macro n(a=g) %}{{ a }}{% endmacro %}"
    "{{ n() }}",
    "{% macro m() %}{{ foo }}{% endmacro %}{% set foo = 1 %}{{ m() }}{% set foo = 2 %}{{ m() }}",
    "{% macro m() %}[{{ i }}]{% endmacro %}{% for i in [1] %}{{ m() }}{% endfor %}",
    "{% macro fact(n) %}{% if n <= 1 %}1{% else %}{{ n }}*{{ fact(n - 1) }}{% endif %}{% endmacro %}{{ fact(4) }}",
    "{% macro m() %}{{ varargs }}{{ kwargs }}{% endmacro %}{{ m(1, 2, k=3) }}",
    "{% macro wrap(tag) %}<{{ tag }}>{{ caller() }}</{{ tag }}>{% endmacro %}{% call wrap('b') %}bold{% endcall %}",
    "{% macro m(x) %}<{{ caller(x) }}>{% endmacro %}{% call(y) m(5) %}y={{ y }}{% endcall %}",
    "{% macro m() %}{% set w = 1 %}{{ w }}{% endmacro %}{{ m() }}{{ w }}",
    "{% macro m() %}x{% endmacro %}{{ m() ~ 'y' }} {{ m() is string }} {{ m }}",
    "{% macro m() %}[{{ caller is defined }}]{% endmacro %}{{ m() }}",
    "{% macro outer() %}{% macro inner() %}in{% endmacro %}{{ inner() }}{% endmacro %}{{ outer() }}",
    "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(1, 2) }}",
    "{% macro m(a) %}{{ a }}{% endmacro %}{{ m(z=2) }}",
    "{% macro m() %}{{ caller() }}{% endmacro %}{{ m() }}",
    # Filters.
    "{{ '  x  '|trim }}|{{ 'xxaxx'|trim('x') }}|{{ ' \u3000x\xa0 '|trim }}|{{ none|trim }}|{{ 5|trim }}",
    "{{ [1, 'é', none, true, 1.0, {'k': [1, 2]}, (1, 'a')]|tojson }}",
    "{{ {'a': [1, {}], 'b': []}|tojson(indent=2) }}",
    "{{ {'b': 1, 'a': {'d': 1, 'c': 2}}|tojson(sort_keys=true, separators=(',', ':')) }}",
    "{{ {'e': '🎉é', 'n': 1.5}|tojson(ensure_ascii=true) }} {{ 'a\\x01\\x7f\\u2028\\t\"\\\\\\x08\\x0c'|tojson }}"
    " {{ {1: 'x', 2.5: 'y', none: 'z', true: 'w'}|tojson }} {{ {'a': none}|tojson(indent=0) }}"
    " {{ [1]|tojson(indent='--') }} {{ 1e16|tojson }} {{ 'x'|tojson(ensure_ascii=false) }}",
    "{{ messages|tojson }} {{ messages[0]|tojson(indent=4) }}",
    "{{ x|tojson }}",
    "{{ 'abc'|length }} {{ [1, 2]|length }} {{ {'a': 1}|count }} {{ 'ʻokina'|length }} {{ x|length }}",
    "{{ 'ABC'|lower }} {{ 'abc'|upper }} {{ 'hello world-foo(bar)'|title }} {{ 'hELLO wORLD'|capitalize }}"
    " {{ 'hello world'.title() }} {{ '1st thing'|title }} {{ '1st thing'.title() }} {{ 'a'.capitalize() }}"
    " {{ none|lower }} {{ 'ʻokina 漢字 🎉 abc'|upper }} {{ 'ʻokina'.title() }} {{ 'aʻb 漢x'|title }}",
    "{{ ['a', 'b']|join(', ') }} {{ [1, none]|join }} {{ messages|join('|', attribute='role') }}"
    " {{ [{'n': 'x'}, {'n': 'y'}]|map(attribute='n')|join('-') }} {{ ['a', 'b']|join(d='+') }}",
    "{{ 'a  b'|replace(' ', '_') }} {{ 'aaa'|replace('a', 'b', 2) }} {{ 'ab'|replace('', '-') }}"
    " {{ 5|replace('5', 'five') }}",
    "{{ x|default('d') }} {{ ''|default('d') }} {{ ''|default('d', true) }} {{ none|default('d') }} {{ x|d('z') }}"
    " {{ 0|d(5, boolean=true) }} {{ x|default }}|",
    "{{ [1, 2]|first }} {{ []|first }} {{ 'ab'|last }} {{ {'a': 1, 'b': 2}|first }} {{ (1, 2)|last }}"
    " {{ 'Abc'|first }} {{ {'a': 1, 'b': 2}|last }}",
    "{{ (1, 2)|list }} {{ 'ab'|list }} {{ {'a': 1}|list }} {{ {'a': 1}|items|list }} {{ x|items|list }}",
    "{{ none|string }} {{ [1, 'a']|string }} {{ 5|string|length }} {{ 'x'|string }}",
    "{{ '3'|int }} {{ '3.7'|int }} {{ 'x'|int }} {{ 3.9|int }} {{ '0x1A'|int(0, 16) }} {{ '2.5'|float }}"
    " {{ 'x'|float }} {{ true|int }} {{ '  12  '|int }} {{ '1_0'|int }} {{ none|int }} {{ '1e3'|int }}"
    " {{ 3|float }} {{ '-2'|int }} {{ 'x'|int(7) }} {{ '0b11'|int(base=0) }} {{ '010'|int(base=0) }}"
    " {{ 'inf'|float }} {{ '-Infinity'|float }} {{ '1_000.5'|float }} {{ '.5'|float }} {{ '5.'|float }}"
    " {{ 'nan'|int }} {{ 'ff'|int(base=16) }}",
    "{{ -3|abs }} {{ 3.7|abs }} {{ false|abs }} {{ -2.5|abs }}",
    "{{ 'a\\nb\\n\\nc'|indent }}|{{ 'a\\nb\\n\\nc'|indent(2, true, true) }}|{{ 'x'|indent(first=true) }}"
    "|{{ 'a\\n'|indent }}|{{ 'a\\nb'|indent('> ') }}|{{ ''|indent(first=true) }}|{{ 'a\\r\\nb\\rc'|indent(1) }}",
    "{{ ['b', 'A', 'a']|unique|list }} {{ ['a', 'A', 'b']|unique(case_sensitive=true)|list }}"
    " {{ [{'n': 'x'}, {'n': 'X'}]|unique(attribute='n')|list }} {{ [1, 1.0, true, 2]|unique|list }}",
    "{{ [1, 2]|reverse|list }} {{ 'abc'|reverse }} {{ [1, 2, 3]|reverse|first }} {{ {'a': 1, 'b': 2}|reverse|list }}"
    " {{ 'ʻ漢🎉'|reverse }}",
    "{{ [3, 1, 2]|sort }} {{ ['b', 'A', 'a']|sort }} {{ [1, 2]|sort(true) }} {{ ['B', 'a']|sort(case_sensitive=true) }}"
    " {{ [{'a': 2}, {'a': 1}]|sort(attribute='a') }} {{ [{'a': 1, 'b': 2}, {'a': 1, 'b': 1}]|sort(attribute='a,b') }}"
    " {{ [2, 1.5, true]|sort }} {{ messages|sort(attribute='role')|map(attribute='role')|list }}"
    " {{ [('b', 1), ('a', 2), ('a', 1)]|sort }} {{ ['x', 'X', 'y']|sort(reverse=true) }}",
    "{{ ['a', 1]|sort }}",
    "{{ messages|map(attribute='role')|list }} {{ [{'x': {'y': 3}}]|map(attribute='x.y')|list }}"
    " {{ [[1, 2]]|map(attribute='0')|list }} {{ ['a ', ' b']|map('trim')|list }} {{ [1, 2]|map('string')|join(',') }}"
    " {{ [{'n': 1}, {}]|map(attribute='n', default=0)|list }} {{ ['a', 'b']|map('upper')|first }}"
    " {{ ['a,b']|map('replace', ',', ';')|list }}",
    "{{ messages|selectattr('role', 'equalto', 'user')|list|length }}"
    " {{ messages|rejectattr('role', 'eq', 'user')|map(attribute='role')|list }}"
    " {{ [{'r': 'u'}, {'x': 'a'}]|selectattr('r')|list }}"
    " {{ [{'r': 'u'}, {'x': 'a'}]|rejectattr('r', 'defined')|list }}"
    " {{ messages|selectattr('content', 'string')|list|length }} {{ messages|selectattr('role', '==', 'user')|first }}",
    "{{ [1, 2, 3, 4]|select('odd')|list }} {{ [1, 2, 3, 4]|reject('even')|list }} {{ [none, 0, 1, '']|select|list }}"
    " {{ [1, 2, 3]|select('greaterthan', 1)|list }} {{ [1, 2, 3]|select('in', [1, 3])|list }}"
    " {{ x|select|list }} {{ [1, 2]|select('ge', 2)|list }}",
    "{{ 'a'|safe + '<' }} {{ '<' + 'a'|safe }} {{ '<b>'|e }} {{ '<'|escape ~ '<' }} {{ ('a'|safe).upper() }}"
    " {{ ('<x>'|e)|length }} {{ '&\\'\"'|e }} {{ none|e }} {{ ('a'|e).replace('a', '<') }} {{ '-'|e }}",
    "{% set it = [1, 2, 3]|select %}{{ it|first }}{{ it|list }}{{ it|list }}{{ 2 in it }}",
    "{% set it = [1, 2, 3]|map('string') %}{{ '1' in it }}{{ it|list }}",
    "{{ [1, 2]|reverse|length }}",
    # Methods.
    "{{ 'Hello World'.split() }} {{ ' a b '.split(' ') }} {{ 'a,b,,c'.split(',', 1) }} {{ 'a b c'.rsplit(' ', 1) }}"
    " {{ 'a b c'.split(maxsplit=1) }} {{ ''.split() }} {{ ''.split(',') }} {{ '\u3000 x\u2028y'.split() }}"
    " {{ '  a b  c  '.split(None, 1) }} {{ '  a b  c  '.rsplit(None, 1) }}"
    " {{ 'a</think>b</think>c'.split('</think>') }}"
    " {{ 'a::b::c'.rsplit('::') }} {{ 'abab'.split('ab') }}",
    "{{ 'abc'.startswith('a') }} {{ 'abc'.endswith('bc') }} {{ 'abc'.startswith(('x', 'a')) }}"
    " {{ 'abc'.startswith('b', 1) }} {{ 'abc'.endswith('a', 0, 1) }} {{ 'abc'.startswith('', 4) }}"
    " {{ 'ʻokina'.startswith('ʻ') }} {{ '漢字'.endswith('字') }} {{ ''.endswith('') }}",
    "{{ '  a  '.strip() }}|{{ 'xyx'.strip('x') }}|{{ '\\n\\nx\\n'.lstrip('\\n') }}|{{ 'x  '.rstrip() }}"
    "|{{ '  x'.lstrip() }}|{{ 'ʻaʻ'.strip('ʻ') }}|{{ 'abc'.strip('') }}|{{ '\\x1cx\\x1f'.strip() }}",
    "{{ 'aaa'.replace('a', 'b') }} {{ 'aaa'.replace('a', 'b', 1) }} {{ 'ab'.replace('', '-') }}"
    " {{ 'ab'.replace('', '-', 2) }} {{ 'a'.replace('a', '') }}|",
    "{{ 'ab'.find('b') }} {{ 'éb'.find('b') }} {{ 'abc'.find('z') }} {{ 'abcabc'.rfind('b') }} {{ 'aaa'.count('a') }}"
    " {{ 'x'.count('') }} {{ 'abc'.count('b', 1) }} {{ 'ʻokina'.find('k') }} {{ 'abc'.find('', 5) }}"
    " {{ 'abc'.find('c', -1) }} {{ 'aaaa'.count('aa') }}",
    "{{ 'Hello'.lower() }} {{ 'hello'.upper() }} {{ 'hello'.capitalize() }} {{ 'hello world'.title() }}"
    " {{ '-'.join(['a', 'b']) }} {{ ''.join('xyz') }} {{ ', '.join(messages|map(attribute='role')) }}",
    "{{ {'k': 1}.get('k') }} {{ {'k': 1}.get('z', 'd') }} {{ {'k': 1}.get('z') }} {{ {'a': 1, 'b': 2}.keys()|list }}"
    " {{ {'a': 1}.values()|list }} {{ {'a': 1, 'b': 2}.items() }} {{ {'a': 1}.keys() }} {{ messages[0].get('role') }}"
    " {{ {'a': 1}.items()|length }} {{ 'a' in {'a': 1}.keys() }}",
    "{{ [1, 2, 1].count(1) }} {{ 'x'.join([1]) if false else 'n' }}",
    "{{ '-'.join([1]) }}",
    # Printing values.
    "{{ none }} {{ true }} {{ false }} {{ True }} {{ 1.5e3 }} {{ 0x10 }} {{ 1_000 }} {{ 0b101 }} {{ 0o17 }}"
    " {{ [1, 'a', none, true, 1.5, {'k': [()]}, (1,), ('a', 2)] }}",
    "{{ [\"it's\", 'say \"hi\"', 'both \\'\"', '\\n\\t\\x00\\u200b\\xa0é🎉\\x7f\\\\'] }}",
    "{{ {'b': 1, 'a': 2} }} {{ messages[0] }} {{ (1, 2) }} {{ () }} {{ 1, 2 }} {{ dict(a=1, b='x') }}"
    " {{ range(3) }} {{ range(1, 9, 2) }} {{ namespace(a=[1]) }} {{ {1: 'a', true: 'b'} }} {{ dict({'z': 1}, y=2) }}",
    "{{ [1.0, 1e20, -0.0, 1e-7, 0.1, 2.5e-310, 1e22] }} {{ 1e300 * 1e10 }} {{ -(1e300 * 1e10) }} {{ 5e-324 }}"
    " {{ 1.7976931348623157e308 }} {{ 123456789012345680.0 }} {{ 9007199254740993.0 }} {{ 0.3 }} {{ 2.0 ** 70 }}",
    # Context variables and globals.
    "{{ add_generation_prompt }} {{ bos_token }} {{ eos_token }} {{ tools }} {{ tools is defined }}"
    " {{ documents is none }} {{ date_string }} {{ enable_thinking }} {{ custom_tools is defined }}",
    "{{ strftime_now is defined }} {{ raise_exception is defined }} {{ range is defined }} {{ namespace is defined }}",
    "{{ raise_exception('custom ' ~ messages|length) }}",
    "{% if messages|length > 0 %}{{ raise_exception('Bad: \"' ~ messages[0].role ~ '\"\\nnext line') }}{% endif %}",
    "{% for m in messages %}{% generation %}[{{ m.content }}]{% set g = 1 %}{% endgeneration %}{{ g }}{% endfor %}",
    "{{ strftime_now('%Y-%m-%d') == strftime_now('%Y-%m-%d') }} {{ strftime_now('%%')|length }}",
]
SYNTHETIC_MESSAGES = [
    {"role": "system", "content": "sys ʻ"},
    {"role": "user", "content": "Hi\nthere"},
    {"role": "assistant", "content": "<think>r</think>ok"},
]
SYNTHETIC_TOKENS = {"bos_token": "<s>", "eos_token": "</s>"}
SYNTHETIC_VARIABLES = {"date_string": "26 Jul 2024", "enable_thinking": False}


@pytest.mark.filterwarnings("ignore:invalid escape sequence:DeprecationWarning")
@pytest.mark.parametrize("index", range(len(SYNTHETIC)))
def test_synthetic_template_matches_jinja2(native_cli, workdir, index):
    source = SYNTHETIC[index]
    expected = oracle(source, SYNTHETIC_MESSAGES, tokens=SYNTHETIC_TOKENS, variables=SYNTHETIC_VARIABLES)
    actual = native(native_cli, workdir, SYNTHETIC_MESSAGES, source=source, tokens=SYNTHETIC_TOKENS,
                    variables=SYNTHETIC_VARIABLES)
    assert_same(actual, expected)


# Python results the renderer does not reproduce: it fails instead of rendering differently.
RENDER_LIMITS = [
    "{{ 9223372036854775807 + 1 }}",  # big integers
    "{{ 'é'|upper }}",  # Unicode case mapping
    "{{ 'Ünïcode'.lower() }}",
    "{{ ['é', 'a']|sort }}",  # case-insensitive keys lower() the strings
    "{{ '%s and %s' % ('a', 'b') }}",  # printf formatting
    "{{ {'items': 1}.items }}",  # printing a bound method
    "{{ [1, 2]|map('string') }}",  # printing a generator
    "{{ '١٢'|int }}",  # non-ASCII digits
    "{{ range(5)[1:] }}",  # range slicing
    "{{ (-8) ** 0.5 }}",  # complex results
]


@pytest.mark.parametrize("source", RENDER_LIMITS)
def test_unreproducible_results_fail_closed(native_cli, workdir, source):
    assert oracle(source, SYNTHETIC_MESSAGES)[0] == "ok"
    actual = native(native_cli, workdir, SYNTHETIC_MESSAGES, source=source)
    assert actual[0] == "error" and "not supported" in actual[1], actual


# --- randomized differential ------------------------------------------------------------

PIECES = [
    "", " ", "\n", "\n\n", "  padded  ", "\t", "Aloha", "ʻokina", "kākou", "漢字", "한국어", "🎉", "👍🏽",
    "e\u0301", "{{ x }}", "{% if true %}", "{#", "'", '"', "\\", "\\n", "\r\n", "<think>", "</think>",
    "<think>\nreason\n</think>\n\n", "/think", "/no_think", "/system_override", "<tool_response>",
    "</tool_response>", "<|im_start|>", "<|eot_id|>", "[INST]", "<|channel|>analysis<|message|>",
    "metadata", "a" * 40, "-", "#", "%",
]


def random_conversation(rng):
    pattern = rng.choice(["alternate", "alternate", "alternate", "system", "free"])
    count = rng.randint(1, 6)
    messages = []
    if pattern == "system":
        messages.append({"role": "system", "content": ""})
    for i in range(count):
        if pattern == "free":
            role = rng.choice(["user", "assistant", "system", "user"])
        else:
            role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": ""})
    for message in messages:
        message["content"] = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 6)))
    return messages


@pytest.mark.parametrize("model", MODELS)
def test_randomized_conversations_match_jinja2(native_cli, workdir, model):
    rng = random.Random(f"lokahi-chat-template-{model}")
    outcomes = set()
    for _ in range(40):
        messages = random_conversation(rng)
        variables = {}
        if rng.random() < 0.5:
            variables["enable_thinking"] = rng.random() < 0.5
        if model == "llama3.2" and rng.random() < 0.5:
            variables["date_string"] = "26 Jul 2024"
        expected = check_model(native_cli, workdir, model, messages, add_generation_prompt=rng.random() < 0.7,
                               variables=variables)
        outcomes.add(expected[0])
    assert "ok" in outcomes
