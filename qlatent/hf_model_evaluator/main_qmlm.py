import os
import argparse
from pipeline.pipeline_executor import InitiatePipeline
from global_variables import *
from huggingface_hub import HfApi
from qpsychometric import *


def main(base_dir):
    hf_api = HfApi()

    fill_mask_models = list(hf_api.list_models(filter="fill-mask"))
    filtered_models = [m for m in fill_mask_models if m.pipeline_tag == "fill-mask"]
    unique_filtered_models = list({m.modelId: m for m in filtered_models}.values())

    all_qmlm_questionnaires = all_psychometrics['QMLM'].get_questions()
    all_qmnli_questionnaires = all_psychometrics['QMNLI'][['ASI', 'BIG5', 'CS', 'GAD7', 'PHQ9', 'SOC']].get_questions()
    all_questionnaires = all_qmlm_questionnaires + all_qmnli_questionnaires

    InitiatePipeline(
        models_info=unique_filtered_models,
        questionnaires_questions_lists=all_questionnaires,
        base_dir=base_dir,
        override_results=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch MLM evaluation of all HuggingFace fill-mask models.")
    parser.add_argument('--base_dir', type=str, default='./model_logs',
                        help='Directory to save results (default: ./model_logs)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace API token (optional, reduces rate limiting)')
    args = parser.parse_args()

    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token

    main(args.base_dir)
