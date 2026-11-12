# vLLM + Ray Serve inference server: a community GitOps pattern by Eugo

This [Validated Pattern](https://validatedpatterns.io/) deploys an OpenAI-compatible LLM server on Red Hat®
OpenShift® through Red Hat OpenShift GitOps. The server is a single pod that runs Ray Serve LLM with the vLLM engine. The cluster builds
its image from this repository, using stock packages only: PyPI wheels on NVIDIA's CUDA UBI 9 image. The pattern
also deploys the pieces the server needs: Node Feature Discovery and the NVIDIA GPU Operator, and Vault with External
Secrets for an optional Hugging Face token. It adds a small JupyterLab whose notebook times the server.

**Community sandbox pattern by Eugo — not produced, supported or validated by Red Hat.**

Eugo, which maintains this pattern, runs an HPC platform that rebuilds open-source native and Python packages,
PyTorch and vLLM among them, from source. This pattern does not use it: it installs stock packages only. More at
[eugo.io](https://eugo.io/).

**The problem it solves.** A team that wants a private, OpenAI-compatible LLM endpoint on its own OpenShift
cluster usually assembles it by hand: GPU enablement, a serving runtime, a model cache, a secret for gated models,
and a way to rebuild the image when the stack changes. This pattern turns all of that into one Git repository.
Fork it, set the model, push, and the cluster converges to a running endpoint. Every Python package is pinned by
hash and the base image by digest, so a rebuild installs the same stack, and every change is a reviewable commit.
It uses upstream open-source components only (vLLM, Ray Serve, PyPI wheels on NVIDIA's CUDA UBI 9 image).

> [!WARNING]
> **Cost.** On AWS, the default install creates **one billable GPU instance: `g6.2xlarge` with one NVIDIA L4**.
> AWS bills it for as long as it runs.
>
> - To stop that, set `global.gpuMachineSet: none` in `values-global.yaml` in your fork, and push the change,
>   before you install.
> - Off AWS, nothing is created.
> - Uninstalling does not delete the GPU machine; see [Uninstall](#uninstall).

- [Architecture](#architecture)
- [What is in the stock stack](#what-is-in-the-stock-stack)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [What to expect on the first install](#what-to-expect-on-the-first-install)
- [Using the API](#using-the-api)
- [The notebook](#the-notebook)
- [Changing the model](#changing-the-model)
- [Uninstall](#uninstall)
- [Support](#support)
- [License](#license)
- [Trademarks](#trademarks)
- [What this pattern was verified against](#what-this-pattern-was-verified-against)

Contributors and coding agents: read [AGENTS.md](AGENTS.md). It covers the repo layout, every deploy path, the rules
and troubleshooting.

## Architecture

```mermaid
flowchart TB
  fork["Your fork of this repository"]
  user(["You, in a browser"])

  subgraph ocp["OpenShift cluster"]
    vpo["Validated Patterns Operator"]
    argo["OpenShift GitOps: Argo CD instance vp-gitops"]
    cg["clustergroup app-of-apps, variant prod"]

    subgraph secrets["Secrets"]
      vault["HashiCorp Vault"]
      eso["External Secrets Operator"]
    end

    subgraph gpu["GPU enablement"]
      nfd["Node Feature Discovery"]
      gpuop["NVIDIA GPU Operator"]
      ms["MachineSet job, AWS only: one g6.2xlarge with one NVIDIA L4"]
    end

    subgraph inference["Namespace vllm-inference"]
      bc["BuildConfig ray-serve-llm"]
      isr["ImageStream ray-serve-llm"]
      dep["Deployment ray-serve-llm: Ray head + Ray Serve + vLLM, one GPU"]
      pvc[("PVC ray-serve-llm-models")]
      svc["Service ray-serve-llm, port 8000"]
    end

    subgraph notebook["Namespace vllm-notebook"]
      nbbuild["BuildConfig and ImageStream notebook"]
      nb["Notebook pod: JupyterLab + oauth-proxy"]
      route["Route with OpenShift login"]
    end
  end

  fork -- "Git" --> argo
  vpo --> argo
  argo --> cg
  cg --> vault
  cg --> eso
  cg --> nfd
  cg --> gpuop
  cg --> ms
  cg --> bc
  cg --> nbbuild
  bc --> isr
  isr --> dep
  vault -- "hftoken" --> eso
  eso -- "Secret huggingface-secret" --> dep
  pvc -- "model weights at /models" --> dep
  svc --> dep
  nbbuild --> nb
  user --> route
  route --> nb
  nb -- "OpenAI-compatible HTTP" --> svc
```

- **The Validated Patterns Operator** installs OpenShift GitOps. It points Argo CD at your fork through the
  clustergroup chart, which creates every app listed in `variants/prod/values-prod.yaml`.
- **The server pod** runs a Ray head, Ray Serve and vLLM. It reports Ready only after the model has loaded.
- **The notebook pod** serves JupyterLab behind the OpenShift oauth-proxy. Only users who may edit the
  `vllm-notebook` namespace (its editors and admins, and cluster admins) can open it. It talks to the server
  through the in-cluster Service.

## What is in the stock stack

| Component | Version | Source |
|---|---|---|
| Ray (`ray[llm]`, Ray Serve LLM) | 2.60.0 | PyPI |
| vLLM | 0.29.0 (the version Ray 2.60.0 pins) | PyPI |
| PyTorch | 2.13.0, CUDA 13.0 wheels | PyPI |
| FlashInfer | 0.6.18 | PyPI |
| CUDA toolkit (`nvcc`, runtime) | 13.0.3 | the base image, plus the pip `cuda-toolkit` 13.0.3 pin |
| Python | 3.12 | UBI 9 packages |
| Base image | `nvcr.io/nvidia/cuda:13.0.3-devel-ubi9`, pinned by digest | NVIDIA, no login needed |
| Default model | `Qwen/Qwen2.5-0.5B-Instruct` at a pinned revision, served as `qwen2.5-0.5b-instruct` | Hugging Face, not gated |

- **Locks.** Python packages are installed from hash-locked files, one per CPU architecture
  (`inference-server/requirements-x86_64.lock` and `requirements-aarch64.lock`), with wheels only. A third lock,
  `requirements-pip.lock`, pins the pip that installs them.
- **Everything comes from PyPI.** No package index outside PyPI is used, so FlashInfer compiles its GPU kernels on
  the node when the server starts. The image does not include the patches Ray applies in its own `rayproject/ray-llm`
  images.
- **Telemetry is off.** Ray and vLLM usage statistics and request prompt logging are turned off.
- **Operators:**
  - Red Hat OpenShift GitOps;
  - the Validated Patterns Operator (community);
  - the Red Hat External Secrets Operator;
  - Node Feature Discovery;
  - the NVIDIA GPU Operator (certified).

  HashiCorp Vault comes from the Validated Patterns chart.

The notebook labels the server it times "Stock vLLM 0.29.0 · Ray 2.60.0 · PyTorch 2.13.0 from PyPI · Python 3.12 ·
NVIDIA CUDA 13.0.3 UBI 9 image".

GPU support: built for the GPU architectures in vLLM's release build list
([v0.29.0 release pipeline](https://github.com/vllm-project/vllm/blob/v0.29.0/.buildkite/release-pipeline.yaml));
on OpenShift we have tested only the NVIDIA L4 (see
[What this pattern was verified against](#what-this-pattern-was-verified-against)). CUDA 13 needs NVIDIA driver
version 580 or newer.

## Prerequisites

- OpenShift 4.18 or newer, and cluster-admin access.
- A StorageClass that provisions volumes dynamically, for the model cache.
- The internal image registry set to `Managed`: the cluster builds both images itself. It starts as `Removed` on
  bare metal and vSphere. To check:
  `oc get configs.imageregistry.operator.openshift.io cluster -o jsonpath='{.spec.managementState}'`
- A GPU node. On AWS the pattern creates one (see the cost note above), except on clusters installed into existing
  subnets, where the job cannot copy the subnet: set `gpuMachineSet: none` there and add the GPU node yourself.
  Elsewhere, add a node with an NVIDIA GPU that the NVIDIA GPU Operator supports.
- On your workstation:
  - podman 4.3 or newer;
  - git;
  - `oc`, logged in to the cluster;
  - if `KUBECONFIG` is set, it must point inside your home directory.
- Your own fork of this repository, with your branch pushed. Argo CD deploys from that fork, not from your disk.

## Quick start

1. Fork this repository on GitHub, then clone your fork:
   ```sh
   git clone https://github.com/<you>/vllm-ray-inference-server.git
   cd vllm-ray-inference-server
   ```
2. Edit `values-global.yaml`:
   - `global.gpuMachineSet`: `create` (one AWS g6.2xlarge) or `none`.
   - `global.model.*`: optional.

   Commit and push:
   ```sh
   git commit -am "Configure my install"
   git push
   ```
3. Optional, needed only for a gated model: create your secrets file and put your Hugging Face token in `hftoken`.
   Keep this file out of git.
   ```sh
   mkdir -p ~/.config/validated-patterns
   cp values-secret.yaml.template \
      ~/.config/validated-patterns/values-secret-vllm-ray-inference-server.yaml
   ```
4. Log in and install:
   ```sh
   oc login --token=<token> --server=<api-url>
   ./pattern.sh make install
   ```
5. Watch it come up:
   ```sh
   oc get applications.argoproj.io -A
   oc get builds -n vllm-inference
   oc get pods -n vllm-inference -w
   ```

[AGENTS.md](AGENTS.md#how-to-deploy) also covers the web-console install, private forks, reloading secrets, and how
updates and image rebuilds work.

**Installed under the old name?** This pattern was named `rhvp-vllm-ray-inference-server` until November 2026,
and its namespaces were `rhvp-inference` and `rhvp-notebook`; they are now `vllm-inference` and `vllm-notebook`. An
install made under the old name keeps working. Merging the rename into its branch does not delete the old
namespaces: Argo CD builds and starts a second server and notebook in the new ones, and the old server keeps its
GPU, so with one GPU the new server waits until you delete the old namespaces. To move the install to the new name
instead:

1. Download any notebook work you want to keep.
2. Uninstall it with `PATTERN_NAME=rhvp-vllm-ray-inference-server ./pattern.sh make uninstall`.
3. Once the uninstall has finished, check for the old namespaces and the old model's claim. Delete what is left;
   this deletes the downloaded model. Then check `oc get pv`, as in [Uninstall](#uninstall), step 3.
   ```sh
   oc get namespace rhvp-inference rhvp-notebook
   oc get pvc -n rhvp-inference
   oc delete namespace rhvp-inference rhvp-notebook
   ```
4. Rename your secrets file from `values-secret-rhvp-vllm-ray-inference-server.yaml` to
   `values-secret-vllm-ray-inference-server.yaml`.
5. Install again with `./pattern.sh make install`. The new install builds both images and downloads the model again.

Until you move it, set `PATTERN_NAME=rhvp-vllm-ray-inference-server` on every `make` target you run on it.
Without it, `load-secrets` can load the repo's template instead of your file and replace your token in Vault with
`none`, and `install` creates a second Pattern next to the old one. [AGENTS.md](AGENTS.md#updating-a-running-install)
has the details, including what to delete after a merge and the one target that ignores `PATTERN_NAME`.

## What to expect on the first install

The first install does a lot of work, in roughly this order:

1. The operators install, and Vault starts.
2. On AWS, the MachineSet job creates the GPU node. The GPU Operator then installs the NVIDIA driver on it.
3. The cluster builds the server image from `inference-server/Containerfile`. This is a large build: it pulls a
   multi-gigabyte CUDA development image and installs several gigabytes of wheels, so the build node needs plenty of
   free ephemeral storage. On an m5.2xlarge node it took 17 to 18 minutes.
4. The server pod stays Pending until a node advertises `nvidia.com/gpu`.
5. On its first start, the server downloads the model into the PVC `ray-serve-llm-models`, where it stays across
   restarts. vLLM and FlashInfer also compile GPU kernels. Those caches are not kept, so every new server pod compiles
   them again.
6. The pod turns Ready only once the model is loaded.

`./pattern.sh make install` waits at most 60 minutes for every Argo CD application to be Synced and Healthy, and
reports a failure if they are not. On a new AWS cluster it finished in 30 minutes, but a first install elsewhere
can outlast that wait. The failure undoes nothing: the pattern is already installed and its secrets loaded. To
recover:

```sh
oc get builds -n vllm-inference                  # is the image build still running, or did it fail?
oc logs -f bc/ray-serve-llm -n vllm-inference    # follow the latest build
oc start-build ray-serve-llm -n vllm-inference --follow   # only if the build failed
oc get pods -n vllm-inference                    # Pending = no GPU yet; Running but not Ready = still loading
./pattern.sh make argo-healthcheck               # re-check once things settle
```

To skip the wait entirely, install with `VP_SKIP_ARGO_HEALTHCHECK=true ./pattern.sh make install`, then run
`./pattern.sh make argo-healthcheck` later.

## Using the API

The server is reachable inside the cluster at `http://ray-serve-llm.vllm-inference.svc:8000`. From your workstation,
forward the port:

```sh
oc port-forward svc/ray-serve-llm 8000:8000 -n vllm-inference
```

In another terminal:

```sh
curl -s http://localhost:8000/v1/models

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 64
      }'
```

The `model` field is `global.model.id`. Any OpenAI-compatible client works with `base_url` set to
`http://localhost:8000/v1`.

**Optional Route.** Set `route.enabled: true` in `charts/all/ray-serve-llm/values.yaml` in your fork and push. The
Route raises the router timeout to 10 minutes. It has no authentication: anyone who can reach the cluster's router
can use the model and the GPU.

## The notebook

Open it through its Route and log in with your OpenShift account. The proxy lets in only users who may update the
Service `notebook` in namespace `vllm-notebook`: that namespace's editors and admins, and cluster admins. Any
other logged-in user gets 403 Forbidden. To let someone in, give them the `edit` role there:
`oc adm policy add-role-to-user edit <user> -n vllm-notebook`.

```sh
oc get route -n vllm-notebook
```

The notebook is `inference_timings.ipynb`. It times the server in four sections:

1. **Server pod start to Ready.** Read from the pod's Kubernetes status, so this section runs only inside the
   cluster. Kubernetes records these times in whole seconds, and the server's probes check for its ready file
   every 2 seconds, so Ready can be recorded up to about 2 seconds after the server is serving.
2. **Time to first token.**
3. **Single-stream decode tokens per second.**
4. **Aggregate tokens per second at 1, 8 and 32 concurrent requests.**

How it measures:

- The prompts are distinct, generated from a fixed seed, and each has the same number of words (so about the same
  number of tokens).
- In sections 2 to 4, a warm-up with different prompts runs before the timed requests.
- Every request uses `temperature=0` and a fixed `max_tokens`.
- Tokens are counted from the server's usage report.

Every section measures this stock server live, in your run, and shows its result in seconds or tokens per second.

The pod copies the notebook folder into a scratch directory each time it starts, so your edits are lost when the
pod restarts. Download anything you want to keep. The notebook is generated by `notebooks/make_notebook.py` and is
committed without outputs.

## Changing the model

1. In `values-global.yaml`, set:
   - `global.model.id`: the name clients use;
   - `global.model.source`: the Hugging Face repo id;
   - `global.model.revision`: the new model's Hugging Face commit, or empty for its latest version. Do not keep the
     old model's commit.
2. Push. Argo CD updates the server's ConfigMap, and the server pod restarts and downloads the new model.
3. For a gated model, load your token first ([Quick start](#quick-start), step 3, then
   `./pattern.sh make load-secrets`).

The model has to fit in the GPU's memory alongside the engine settings in the `ray-serve-llm` chart.

## Uninstall

1. Run `./pattern.sh make uninstall`. The Validated Patterns framework marks it experimental; see
   [the uninstall announcement](https://validatedpatterns.io/blog/2026-02-16-pattern-uninstall/). The Validated
   Patterns Operator and OpenShift GitOps stay installed.
2. **Delete the GPU MachineSet yourself, or AWS keeps billing for the instance.** An imperative job created it, not
   Argo CD, so uninstall does not remove it. Do this only after the uninstall, or after `gpuMachineSet: none` has
   synced; otherwise the job may create it again.
   ```sh
   oc get machinesets -n openshift-machine-api \
     -o custom-columns=NAME:.metadata.name,TYPE:.spec.template.spec.providerSpec.value.instanceType
   oc delete machineset <name-of-the-g6.2xlarge-machineset> -n openshift-machine-api
   ```
3. Check `oc get pv` for a leftover volume from the claim `vllm-inference/ray-serve-llm-models`.

## Support

Support is community best effort, through
[GitHub issues](https://github.com/eugo-inc/vllm-ray-inference-server/issues). No response time is promised.
Red Hat does not support this pattern.

## License

This repository is licensed under [Apache-2.0](LICENSE). Server code ported from Eugo is licensed under
BSD-3-Clause, Copyright (c) 2026, Eugo; those files say so in their header (see [LICENSES/](LICENSES/)). Packages
installed at build time keep their own licenses.

## Trademarks

Red Hat, Red Hat OpenShift and OpenShift are trademarks or registered trademarks of Red Hat, LLC or its
subsidiaries in the United States and other countries. vLLM and Ray are trademarks of LF Projects, LLC. PyTorch is
a trademark of The Linux Foundation. NVIDIA and CUDA are trademarks or registered trademarks of NVIDIA
Corporation. Other names are trademarks of their respective owners. They are used here only to identify the
software this pattern deploys, and their use does not imply endorsement.

## What this pattern was verified against

The installs below ran before the pattern and its namespaces were renamed, under its old name
`rhvp-vllm-ray-inference-server`, which shows below where the installer used it.

- **A fresh install on AWS** (November 2026): a new installer-provisioned compact cluster, not changed after the
  installer ran: OpenShift 4.22.18, x86_64, region us-east-1, three m5.2xlarge nodes that are both control plane and
  workers. The pattern was installed from `main` at commit `1a781d1` with the default settings in
  `values-global.yaml`.
  - `./pattern.sh make install` installed the Validated Patterns Operator 0.0.82, Red Hat OpenShift GitOps 1.21.7,
    the External Secrets Operator 1.2.1, Node Feature Discovery 4.22 and the NVIDIA GPU Operator 26.7.2. It finished
    in 30 minutes, inside its 60-minute wait, with all seven Argo CD applications (the six apps plus the clustergroup
    app `rhvp-vllm-ray-inference-server-prod`) Synced and Healthy.
  - The pattern's MachineSet job created the g6.2xlarge node with one NVIDIA L4, and the GPU Operator made it
    advertise `nvidia.com/gpu: 1`.
  - The cluster built both images itself: the x86_64 server image in about 18 minutes, the notebook image in under 2.
  - The server downloaded the model, turned Ready without a restart, and answered `GET /v1/models` and a chat
    completion through its Service. The notebook pod ran behind oauth-proxy.
  - The sample notebook ran headless inside the notebook pod against the live server, with no errors. All four
    sections produced results, the first one read from the server pod's Kubernetes status. That run used the
    notebook as it was then, with an empty second side in sections 1 and 2 that has since been removed; the code
    that measures is unchanged.
- **The AWS install before it found a bug** that the earlier checks did not: OpenShift's in-cluster builder records
  an empty `CMD []` as `/bin/sh -c null`, so the server received those words as arguments and exited. Docker builds
  do not do this, and on OpenShift Local the server pod never started, for lack of a GPU. Both Containerfiles and the
  server chart were fixed, and the tests now check for it. On that cluster, after the fix was pushed, the rebuild job
  rebuilt both images by itself, and the server came up on the rebuilt image.
- **An install on OpenShift Local** (OpenShift 4.22.14, arm64, no GPU), from a test branch that set `arch: arm64` in
  both charts: the operators, Vault and External Secrets, both in-cluster image builds (the arm64 server image in
  about 14 minutes) and the notebook pod behind its OpenShift login came up. Without a GPU the server pod stayed
  Pending, so nothing was served.
- **The aarch64 server image with docker on one NVIDIA GPU**, running as an arbitrary UID in group 0: the model
  loaded, a chat request was answered, and SIGTERM stopped the server cleanly.
- **On every change** (the tests in `tests/` and the CI workflows): the charts are rendered with Helm and
  schema-checked against the clustergroup chart's values schema, and the secrets template is checked against the
  Validated Patterns secrets schema.
- **Not checked:** OpenShift versions before 4.22; a gated model with a real Hugging Face token; the optional server
  Route; changing the model; `image.override`; an arm64 GPU node on OpenShift; other clouds and bare metal; GPUs
  other than the L4 (see [GPU support](AGENTS.md#gpu-support)); the web-console install; private forks;
  `./pattern.sh make uninstall`.
