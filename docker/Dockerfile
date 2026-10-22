FROM vllm/vllm-openai:v0.15.1
# The base image owns the CUDA/PyTorch/vLLM stack; do not pip-upgrade it here.
WORKDIR /workspace
COPY scripts/serve.sh /workspace/scripts/serve.sh
COPY configs/experiment.json /workspace/configs/experiment.json
COPY src /workspace/src
COPY pyproject.toml /workspace/pyproject.toml
ENTRYPOINT ["bash", "/workspace/scripts/serve.sh"]
CMD ["fp16"]
