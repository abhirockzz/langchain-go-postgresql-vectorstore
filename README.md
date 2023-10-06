# PostgreSQL Vector Database implementation for LangChain Go

[langchaingo](https://github.com/tmc/langchaingo) extension to use [pgvector](https://github.com/pgvector/pgvector) as a vector database for your Go applications. It uses the [pgvector-go](https://github.com/pgvector/pgvector-go) library along with [pgx](https://github.com/jackc/pgx) driver.

You can use this in your LangChain applications as a standalone vector database or more likely, as part of a chain. For example, in a RAG implementation:

```go
    import(
        "github.com/abhirockzz/langchain-go-postgresql-vectorstore/pgvector"
         //...
    )
    func ragToRiches(){
        
        bedrockClaudeLLM, err := claude.New("us-east-1")

        tableName := "test_table"
        textColumnName := "text_data"
        embeddingStoreColumnName := "embedding_data"

        amazonTitanEmbedder, err := titan_embedding.New("us-east-1")

        pgVectorStore, err := pgvector.New(pgConnString, 
                                        tableName, 
                                        embeddingStoreColumnName, 
                                        textColumnName, 
                                        false, 
                                        amazonTitanEmbedder)

        result, err := chains.Run(
            context.Background(),
            chains.NewRetrievalQAFromLLM(
                bedrockClaudeLLM,
                vectorstores.ToRetriever(pgVectorStore, numOfResults),
            ),
            question,
            chains.WithMaxTokens(8091),
        )
    }
```