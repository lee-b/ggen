import argparse
import json
import logging
import sys
from pathlib import Path
from importlib import resources

from llama_cpp.llama import Llama, LlamaGrammar


logger = logging.getLogger(__name__)


class Config:
    """
    Config class to manage configuration for the generative AI tool.
    """
    def __init__(self):
        # Defaults
        self.model_name = None
        self.model_path = None
        self.schema_name = None
        self.schema_path = None
        self.prompt_template_file = None
        self.prompt_template_path = None

        # Defaults for search paths
        self.model_paths = []
        self.schema_paths = []
        self.prompt_template_paths = []

        # Load defaults from standard directories
        self.load_defaults_from_standard_directories()

    def load_defaults_from_standard_directories(self):
        # Using importlib.resources to access the package's resources directory
        with resources.path(__package__, 'resources') as resources_path:
            self._add_resources_from_directory(resources_path)

        # Standard directories
        standard_dirs = [
            "/etc/ggrama",
            "/usr/share/ggrama",
            "/usr/local/share/ggrama",
            "~/.local/ggrama",
            "~/.config/gramma"
        ]

        for d in standard_dirs:
            expanded_dir = Path(d).expanduser()
            self._add_resources_from_directory(expanded_dir)

    def _add_resources_from_directory(self, directory):
        if directory.is_dir():
            for subdir in ["models", "schemas", "prompt_templates"]:
                path = directory / subdir
                if path.is_dir():
                    # Correct the attribute names here
                    if subdir == "models":
                        self.model_paths.append(path)
                    elif subdir == "schemas":
                        self.schema_paths.append(path)
                    elif subdir == "prompt_templates":
                        self.prompt_template_paths.append(path)

    def load_from_file(self, config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)


def build_argparser():
    parser = argparse.ArgumentParser(description="Generative AI command-line tool. This tool generates output "
                                                 "based on given input prompts, templates, and AI models. It supports "
                                                 "loading prompts from files or standard input and can use specified "
                                                 "or default paths to locate models and templates.")
    parser.add_argument("--config-file", "-c", action="append", default=[],
                        help="Path to a JSON configuration file. Multiple configuration files can be specified. "
                             "Each file should be in JSON format with keys such as 'model_paths', 'schema_paths', "
                             "'prompt_template_paths', 'model_name', 'model_path', 'schema_name', 'schema_path', "
                             "'prompt_template_file', and 'prompt_template_path'. Each key should map to a string or "
                             "a list of strings representing paths or names. Files are loaded in the order they are specified.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging for detailed output.")
    parser.add_argument("--input", "-i", dest="input_file", type=str,
                        help="Path to a file containing the prompt.")
    parser.add_argument("--output", "-o", dest="output_file", type=str,
                        help="Path to the output file where the result will be saved. If not specified, output is "
                             "printed to standard output.")
    parser.add_argument("--model-name", "-m", dest="model_name", type=str,
                        help="Name of the model to be used for generation. The model must be located in one of the "
                             "configured model paths.")
    parser.add_argument("--model-path", "-M", dest="model_path", type=str,
                        help="Direct path to the directory containing the AI model. Overrides model path configurations.")
    parser.add_argument("--schema-name", "-s", dest="schema_name", type=str,
                        help="Name of the schema file to be used. The schema defines the grammar for the AI output.")
    parser.add_argument("--schema-path", "-S", dest="schema_path", type=str,
                        help="Direct path to the directory containing the schema file. Overrides schema path configurations.")
    parser.add_argument("--prompt-template", "-t", dest="prompt_template_file", type=str, default='instruct',
                        help="Name of the prompt template file. The template structures the input prompt. Default: 'instruct'.")
    parser.add_argument("--prompt-template-path", "-T", dest="prompt_template_path", type=str,
                        help="Direct path to the directory containing prompt templates. Overrides prompt template path configurations.")
    parser.add_argument("prompt", nargs="*", help="Direct input prompt. Used if no prompt file is specified. Multiple arguments are concatenated.")

    return parser


def load_prompt_template(prompt_template_paths, prompt_template_name):
    if prompt_template_name is None:
        raise ValueError("No prompt template name provided")

    for path in prompt_template_paths:
        template_path = Path(path) / f"{prompt_template_name}.txt"

        if template_path.exists():
            with open(template_path, 'r') as f:
                return f.read()

    raise FileNotFoundError(f"Prompt template {prompt_template_name}.txt not found in paths {prompt_template_paths}")


def find_model_path(model_paths, model_name):
    if model_name is None:
        raise ValueError("No model name provided")
    for path in model_paths:
        model_path = Path(path) / model_name
        if model_path.exists():
            return model_path
    raise FileNotFoundError(f"Model {model_name} not found in paths {model_paths}")


def load_schema(schema_paths, schema_name):
    if schema_name is None:
        raise ValueError("No schema name provided")

    for path in schema_paths:
        # First, try to find the schema with the exact name provided
        schema_path = Path(path) / schema_name
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                return json.load(f)

        # If not found, try with the '.gbnf' extension
        schema_path_with_extension = Path(path) / f"{schema_name}.gbnf"
        if schema_path_with_extension.exists():
            with open(schema_path_with_extension, 'r') as f:
                return f.read()

    raise FileNotFoundError(f"Schema {schema_name} not found in paths {schema_paths} and no .gbnf extension file found")


def generate_with_grammar(model, prompt_template, prompt, input_data, grammar):
    full_prompt = prompt_template.format(prompt=prompt, input_data=input_data)
    response = model(full_prompt, grammar=grammar, max_tokens=-1)
    result = json.loads(response['choices'][0]['text'])
    return result


def main():
    parser = build_argparser()
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # Initialize and configure the Config object
    config = Config()
    for config_file in args.config_file:
        config.load_from_file(config_file)

    # Override config with command-line arguments if provided
    model_name = args.model_name if args.model_name else config.model_name
    model_path = args.model_path if args.model_path else config.model_path
    schema_name = args.schema_name if args.schema_name else config.schema_name
    schema_path = args.schema_path if args.schema_path else config.schema_path
    prompt_template_file = args.prompt_template_file if args.prompt_template_file else config.prompt_template_file
    prompt_template_path = args.prompt_template_path if args.prompt_template_path else config.prompt_template_path

    # Read prompt from stdin, input file, or direct argument
    if args.input_file:
        with open(args.input_file, 'r') as file:
            input_data = file.read()
    else:
        input_data = ''

    prompt = ' '.join(args.prompt)
    prompt = prompt.strip()

    if prompt == "":
        logger.fatal("No prompt given!")
        return 1

    # Load resources based on configuration and command-line arguments
    prompt_template = load_prompt_template(config.prompt_template_paths + ([prompt_template_path] if prompt_template_path else []), prompt_template_file)
    model_path = find_model_path(config.model_paths + ([model_path] if model_path else []), model_name)
    schema = load_schema(config.schema_paths + ([schema_path] if schema_path else []), schema_name)

    # Validate critical components
    if not schema:
        logger.error("Schema not provided or not found")
        return 1

    # Initialize the Llama model and grammar
    grammar = LlamaGrammar.from_string(schema, verbose=False)
    llm = Llama(str(model_path), verbose=False)

    # Generate output with grammar
    result = generate_with_grammar(llm, prompt_template, prompt, input_data, grammar)

    # Write result to output file or stdout
    if args.output_file:
        with open(args.output_file, 'w') as file:
            json.dump(result, file)
    else:
        print(json.dumps(result, indent=4))

    return 0
