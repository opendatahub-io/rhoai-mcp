# AutoML MCP deployments

The AutoML domain exposes deployed AutoGluon tabular models to MCP clients. It does
not expose every KServe `InferenceService`: an administrator or project user must
explicitly opt a deployment into MCP access.

## Expose a deployment

An eligible `InferenceService` must use the `autogluon` model format and have this
label:

```yaml
metadata:
  labels:
    automl.opendatahub.io/mcp-exposed: "true"
```

It also needs the `automl.opendatahub.io/input-schema` annotation. Its value is a
JSON document that describes the feature names and JSON types the predictor accepts.
For a binary classification model, it can look like this:

```yaml
metadata:
  annotations:
    automl.opendatahub.io/input-schema: >-
      {"schema_version":"v1","task_type":"classification","features":[{"name":"age","json_type":"integer"},{"name":"country","json_type":"string"}],"example_record":{"age":42,"country":"PL"},"class_labels":["no","yes"]}
```

The schema is consumed by the MCP server, which validates requests before it calls
the model. A future AutoGluon deployment flow can generate this annotation from its
training metadata; it is deliberately versioned so that format can evolve without
changing the MCP tool contract.

For classification probability output, configure the AutoGluon `ServingRuntime`
with:

```yaml
spec:
  containers:
    - env:
        - name: PREDICT_PROBA
          value: "true"
```

Without that setting, the model server returns only the predicted class. The MCP
tool returns that class and explicitly reports that probabilities are unavailable.

## Available tools

The AutoML plugin provides three namespace-scoped, Kubernetes-RBAC-aware tools:

| Tool | Purpose |
| --- | --- |
| `list_automl_deployments(namespace)` | Lists explicitly exposed and AutoGluon-backed deployments. |
| `get_automl_deployment(name, namespace)` | Returns deployment status, input schema, endpoint, and example record. |
| `predict_automl_deployment(name, namespace, records)` | Validates one or more JSON records, calls the KServe v2 endpoint, and returns structured results. |

`records` is a small batch of JSON objects. Every object must contain each required
feature with the JSON type declared in the schema. For example:

```json
{
  "name": "churn",
  "namespace": "demo",
  "records": [
    {"age": 42, "country": "PL"}
  ]
}
```

For a classification deployment configured with `PREDICT_PROBA=true`, a result has
one entry per record:

```json
{
  "task_type": "classification",
  "probabilities_available": true,
  "predictions": [
    {
      "class": "yes",
      "probabilities": {"no": 0.1, "yes": 0.9}
    }
  ]
}
```

Regression predictions use `{"value": 123.4}` instead. Validation and endpoint
failures are returned as concise errors, such as
`records[0] is missing required feature 'country'`.

## Project Navigator and other MCP clients

When the AutoML plugin is enabled, the MCP tool list and each tool description tell
an MCP-capable agent what it can do. A typical agent flow is: list deployments, get
the selected deployment to read its schema/example, then call the predict tool.

To use the server from Project Navigator, deploy the MCP server behind the
platform's supported authenticated MCP endpoint and register that endpoint in the
platform-level MCP server configuration. The server's Kubernetes ServiceAccount
must have `get` and `list` access to `serving.kserve.io/inferenceservices` in each
allowed namespace. No external API key is used by this plugin.
