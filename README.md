<p align="center">
<a href="https://www.linkedin.com/in/soriamaximilianorodrigo/" target="_blank" rel="noopener noreferrer">
<img width="100%" height="100%" src="docs/img/banner.gif" alt="ia-07-embeddings-service"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square" alt="Python-3.11+">
  <img src="https://img.shields.io/badge/FastAPI-OpenAI__compatible-14B8A6?style=flat-square" alt="FastAPI-OpenAI__compatible">
  <img src="https://img.shields.io/badge/embeddings-BGE%2FE5-22D3EE?style=flat-square" alt="embeddings-BGE%2FE5">
  <img src="https://img.shields.io/badge/tests-pytest-1DE9B6?style=flat-square" alt="tests-pytest">
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=1DE9B6&center=true&vCenter=true&width=820&lines=embeddings+self-hosted+%C2%B7+OpenAI-compatible;%2Fv1%2Fembeddings+%C2%B7+%2Fv1%2Frerank+%C2%B7+%2Fhealth;el+dato+no+sale+de+tu+red+%28GDPR%29" alt="typing SVG">
</p>

<hr/>

<h1 align="center">ia-07-embeddings-service</h1>

<p align="center">
Servicio de <b>embeddings self-hosted</b> con API <b>OpenAI-compatible</b> (FastAPI): la pieza fundacional que consumen el resto de los proyectos.
</p>

## ¿Qué resuelve este proyecto?

Enviar código, logs o esquemas propietarios a una API de embeddings externa choca con privacidad/GDPR; el fallback por hashing degrada la calidad. Este microservicio expone embeddings (y reranking) con una **API OpenAI-compatible**, para que todo el ecosistema lo consuma sin cambiar clientes, manteniendo **el dato dentro de tu red**. Es la pieza **fundacional** de la capa de IA.

## ¿Qué pasos sigue?

Clientes idénticos a OpenAI pegan a `/v1/embeddings`; por defecto sirve con un embedder offline, y en producción se reemplaza por BGE/E5 en GPU manteniendo el mismo contrato.

```mermaid
flowchart LR
    C[Clientes: IA-01/03/05/06] -->|/v1/embeddings| API[FastAPI OpenAI-compatible]
    API --> M[(Modelo BGE/E5 · GPU)]
    API --> H[/health · /v1/rerank/]
```

## Componentes principales

- **`create_app`** — app FastAPI con `/v1/embeddings`, `/v1/rerank`, `/health`.
- **`ia_core.HashingEmbedder`** — backend offline por defecto (sin GPU).
- **Contrato OpenAI-compatible** — se cambia el modelo real sin tocar clientes.

## ¿Por qué así?

Mantener el contrato OpenAI significa que `mk5-toolkit` y los demás no cambian de cliente al pasar de la nube a local. Con el servicio local, el corpus **nunca sale de la red** — base para que los agentes corran 100% privados.

## Uso

```bash
pip install -e ".[dev]"
pytest -q
```

> Parte del portafolio de **Maximiliano Rodrigo Soria** — capa de IA sobre el ecosistema
> de 9 backends. Corre **offline** (embedder por hashing + LLM stub deterministas);
> en producción se cambian los adaptadores por los reales (IA-07 local / API OpenAI-compatible)
> por los mismos puertos. Diseño y contrato completos en [`TASK.md`](./TASK.md).
