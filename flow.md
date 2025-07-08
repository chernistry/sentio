# Sense Education – Platform Architecture & Data Flow

Below are two diagrams capturing high level platform architecture and the ML/LLM feedback pipeline overview.

---

```mermaid
flowchart LR
    %% === Client Interfaces ===
    subgraph client_layer["Educators & Learners"]
        LTI["LMS / LTI Widget"]
        WebApp["Sense Web App (React / Next.js)"]
        IDE["In-IDE Plugin"]
    end

    %% === Edge & Auth ===
    subgraph edge_auth["Edge & Auth"]
        CDN{"Azure Front Door / CDN"}
        AuthZ["Auth0 / Entra ID"]
    end

    %% === API Tier ===
    subgraph api_tier["API Tier"]
        APIGW["Azure API Management"]
        FastAPI["Python FastAPI Micro-services"]
    end

    %% === Processing & Orchestration ===
    subgraph processing["Processing & Orchestration"]
        EventHub["Azure Event Hub"]
        AKS["AKS - ML & Service Pods"]
        Prefect["Prefect 3 (Orchestration)"]
        Spark["Azure Databricks / Spark 3.5"]
    end

    %% === Data Layer ===
    subgraph data_layer["Data Layer"]
        ADLS["Azure Data Lake Gen 3"]
        Cosmos["Cosmos DB (metadata)"]
        Redis["Azure Cache for Redis"]
        VectorDB["Azure Cognitive Search / Vector Index"]
    end

    %% === ML & LLM Services ===
    subgraph ml_services["ML & LLM Services"]
        AOAI["Azure OpenAI (LLM)"]
        HF["HuggingFace Endpoints"]
        RAG["RAG Micro-service"]
    end

    %% === Analytics ===
    subgraph analytics["Analytics & BI"]
        Synapse["Azure Synapse"]
        PowerBI["Power BI Embedded"]
    end

    %% === DevOps ===
    subgraph devops["DevOps & SecOps"]
        GH["GitHub Actions CI/CD"]
        Bicep["Azure Bicep IaC"]
        Monitor["Azure Monitor + Grafana"]
        Vault["Azure Key Vault"]
    end

    %% ---- Connections ----
    LTI -- HTTPS --> CDN
    WebApp -- HTTPS --> CDN
    IDE -- gRPC --> APIGW

    CDN --> APIGW
    APIGW -->|JWT| AuthZ
    APIGW --> FastAPI

    FastAPI --> EventHub
    FastAPI --> Cosmos
    FastAPI --> Redis

    EventHub --> Prefect
    Prefect --> Spark
    Prefect --> AKS

    Spark --> ADLS
    AKS --> ADLS
    AKS --> VectorDB
    AKS --> AOAI
    AKS --> HF

    AOAI --> RAG
    VectorDB --> RAG
    RAG --> FastAPI

    ADLS --> Synapse
    Synapse --> PowerBI

    GH --> AKS
    GH --> Bicep
    Monitor --> AKS
    Vault --> AKS
```

---
### Legend & Notes
* External traffic terminates at Azure Front Door (TLS + WAF).
* Prefect orchestrates event-driven pipelines; Spark handles batch clustering/analytics.
* RAG blends vector search with LLM for pedagogy-aligned feedback.

---

## ML & LLM Feedback Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant S as Student Submission
    participant LTI as LMS / LTI
    participant API as FastAPI Service
    participant EH as Event Hub
    participant PF as Prefect Flow
    participant SP as Spark Cluster
    participant V as Vector DB
    participant LLM as LLM
    participant R as RAG Service
    participant EDU as Educator Dashboard

    S->>LTI: upload (code / essay)
    LTI->>API: POST /submissions
    API->>EH: enqueue payload
    EH->>PF: trigger flow
    PF->>SP: batch clustering
    SP->>V: store embeddings + clusters
    PF-->>API: status processed
    API->>R: generate feedback(id)
    R->>V: similarity search
    R->>LLM: prompt {context, rubric}
    LLM-->>R: feedback + confidence
    R-->>API: JSON
    API-->>EDU: grouped feedback UI
    EDU-->>S: individual feedback released

    Note over SP: Pattern discovery & rubric alignment
```

