# RHOAI MCP Training

Training component for RHOAI MCP providing Kubeflow Training Operator integration.

## Features

- Training job discovery and management
- Progress monitoring via trainer annotations
- Lifecycle management (suspend, resume, delete)
- Resource planning and estimation
- Training runtime management
- Storage setup for distributed training

## Installation

```bash
pip install rhoai-mcp-training
```

For optional Kubeflow SDK integration:

```bash
pip install rhoai-mcp-training[kubeflow]
```

## Usage

This package is automatically discovered as an RHOAI MCP plugin.

## CRDs

This plugin requires the following CRDs:
- `TrainJob` (trainer.kubeflow.org/v1)
- `ClusterTrainingRuntime` (trainer.kubeflow.org/v1)
