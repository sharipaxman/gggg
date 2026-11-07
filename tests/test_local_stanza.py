"""Локальный адаптер Argos/Stanza: без импорта ML и без сетевых запросов."""
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import MagicMock, patch

from app.local_stanza import LocalStanza, configure_local_stanza
from app.translator import Translator


class LocalStanzaTests(unittest.TestCase):
    def test_disables_download_and_reuses_pipeline(self):
        module = SimpleNamespace(Pipeline=MagicMock(), __version__="test")
        adapter = LocalStanza(module)
        first = adapter.Pipeline(lang="en", dir="local/en", processors="tokenize")
        self.assertIs(first, adapter.Pipeline(lang="en", dir="local/en", processors="tokenize"))
        module.Pipeline.assert_called_once_with(lang="en", dir="local/en", processors="tokenize",
                                               download_method=None)
        self.assertEqual(adapter.__version__, "test")

    def test_separate_languages_and_no_caller_override(self):
        module = SimpleNamespace(Pipeline=MagicMock())
        adapter = LocalStanza(module)
        adapter.Pipeline(lang="en", dir="en", download_method="download")
        adapter.Pipeline(lang="ru", dir="ru")
        self.assertEqual(module.Pipeline.call_count, 2)
        for call in module.Pipeline.call_args_list:
            self.assertIsNone(call.kwargs["download_method"])

    def test_missing_local_resource_is_error_without_retry_online(self):
        module = SimpleNamespace(Pipeline=MagicMock(side_effect=FileNotFoundError("missing tokenizer")))
        with self.assertRaises(FileNotFoundError):
            LocalStanza(module).Pipeline(lang="en", dir="en")
        module.Pipeline.assert_called_once_with(lang="en", dir="en", download_method=None)

    def test_supports_old_and_new_argos_without_patching_stanza_module(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                original_factory = MagicMock()
                module = SimpleNamespace(Pipeline=original_factory)
                dependency = SimpleNamespace(stanza=module)
                argos = dependency if legacy else SimpleNamespace(sbd=dependency)
                configure_local_stanza(argos)
                adapter = dependency.stanza
                configure_local_stanza(argos)
                self.assertIs(dependency.stanza, adapter)
                self.assertIsInstance(adapter, LocalStanza)
                self.assertIs(module.Pipeline, original_factory)

    def test_argos_forces_local_backend_and_warms_both_directions(self):
        package = ModuleType("argostranslate")
        package.__path__ = []
        settings = ModuleType("argostranslate.settings")
        settings.ModelProvider = SimpleNamespace(OPENNMT="local")
        settings.model_provider = "cloud"
        translate = ModuleType("argostranslate.translate")
        en_ru, ru_en = MagicMock(), MagicMock()
        en = SimpleNamespace(code="en", get_translation=lambda _: en_ru)
        ru = SimpleNamespace(code="ru", get_translation=lambda _: ru_en)
        translate.get_installed_languages = MagicMock(return_value=[en, ru])
        package.settings, package.translate = settings, translate
        modules = {"argostranslate": package, "argostranslate.settings": settings,
                   "argostranslate.translate": translate}
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {"ARGOS_MODEL_PROVIDER": "OPENAI"}):
            translator = Translator()
            translator.warm_up()
            self.assertEqual(settings.model_provider, "local")
        en_ru.translate.assert_called_once_with("Hello.")
        ru_en.translate.assert_called_once_with("Привет.")


if __name__ == "__main__":
    unittest.main()
