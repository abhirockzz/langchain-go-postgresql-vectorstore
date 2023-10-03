package pgvector

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	pgv "github.com/pgvector/pgvector-go"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

// PgVectorStore is a wrapper.
type PgVectorStore struct {
	embedder                 embeddings.Embedder
	pool                     *pgxpool.Pool
	tableName                string
	embeddingColumnName      string //name of the column whose value will be embedded
	embeddingStoreColumnName string //name of the column in which embedded vector data will be stored

	// attributes for similarity search
	searchKey       string   // name of the column whose value needs to returned by search
	queryAttributes []string //optional
}

func New(pgConnectionString, tableName, embeddingStoreColumnName string, embedder embeddings.Embedder) (PgVectorStore, error) {
	//connection string example - postgres://postgres:postgres@localhost/postgres
	pool, err := pgxpool.New(context.Background(), pgConnectionString)

	if err != nil {
		return PgVectorStore{}, err
	}

	return PgVectorStore{embedder: embedder, tableName: tableName, embeddingStoreColumnName: embeddingStoreColumnName, pool: pool}, nil
}

var ErrEmbedderWrongNumberVectors = errors.New(
	"number of vectors from embedder does not match number of documents",
)

func (store PgVectorStore) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) error {
	//opts := store.getOptions(options...)
	//nameSpace := store.getNameSpace(opts)

	fmt.Println("ENTER PgVectorStore/AddDocuments")

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}

	fmt.Println("following words will be added to the vector store")
	for _, text := range texts {
		fmt.Println(text)
	}

	vectors, err := store.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return err
	}

	if len(vectors) != len(docs) {
		return ErrEmbedderWrongNumberVectors
	}

	metadatas := make([]map[string]any, 0, len(docs))

	for i := 0; i < len(docs); i++ {
		metadata := make(map[string]any, len(docs[i].Metadata))
		for key, value := range docs[i].Metadata {
			metadata[key] = value
		}

		metadatas = append(metadatas, metadata)
	}

	for i, doc := range docs {
		embedding := convertVector(vectors[i])
		pgVec := pgv.NewVector(embedding)
		metadata := metadatas[i]

		query, values := store.generateInsertQueryWithValues(pgVec, metadata, doc.PageContent)
		fmt.Println("generated query:", query)

		_, err := store.pool.Exec(context.Background(), query, values...)
		if err != nil {
			return err
		}

		fmt.Println("added ")
	}

	fmt.Println("EXIT PgVectorStore/AddDocuments")

	return nil
}

func (store PgVectorStore) generateInsertQueryWithValues(pgVec pgv.Vector, data map[string]any, stringBeingEmbedded string) (string, []any) {

	// Generate column names and placeholders dynamically
	var columns []string
	var placeholders []string
	var values []any

	data[store.embeddingStoreColumnName] = pgVec
	data[store.embeddingColumnName] = stringBeingEmbedded

	for column, value := range data {
		columns = append(columns, column)
		placeholders = append(placeholders, fmt.Sprintf("$%d", len(placeholders)+1))
		values = append(values, value)
	}

	// Construct the dynamic SQL query
	sqlQuery := fmt.Sprintf(
		"INSERT INTO %s (%s) VALUES (%s)",
		store.tableName,
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", "),
	)

	return sqlQuery, values
}

func (store PgVectorStore) SimilaritySearch(ctx context.Context, searchString string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {

	fmt.Println("similarity search for", searchString, "with max docs", numDocuments)

	vector, err := store.embedder.EmbedQuery(ctx, searchString)
	if err != nil {
		return nil, err
	}

	fmt.Println("query embeddeding done")

	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}

	query := store.generateSelectQuery(numDocuments, opts.ScoreThreshold)

	rows, err := store.pool.Query(context.Background(), query, pgv.NewVector(convertVector(vector)))

	if err != nil {
		return nil, err
	}

	fmt.Println("found rows")

	defer rows.Close()

	docs := []schema.Document{}
	doc := schema.Document{}

	for rows.Next() {

		fmt.Println("checking results")

		vals, err := rows.Values()

		if err != nil {
			return nil, err
		}

		fmt.Println("no. vals in this row", len(vals))

		doc.PageContent = vals[0].(string)

		fmt.Println("doc page content -", doc.PageContent)

		metadata := make(map[string]any)
		metadata["similarity_score"] = vals[1]

		for i := 2; i <= len(vals)-1; i++ {
			metadata[store.queryAttributes[i-2]] = vals[i]
			fmt.Println("metadta -", metadata)
		}

		doc.Metadata = metadata

		docs = append(docs, doc)
	}

	return docs, nil
}

func (store PgVectorStore) generateSelectQuery(numDocuments int, threshold float64) string {

	//"select question, answer from pgx_items where 1 - (q_embedding <=> $1) > 0 LIMIT 2"

	//sqlQuery := fmt.Sprintf("SELECT %s, %s FROM %s WHERE 1 - (%s <=> $1) > %v LIMIT %d", store.searchKey, strings.Join(store.queryAttributes, ","), store.tableName, store.embeddingStoreColumnName, threshold, numDocuments)

	sqlQuery := fmt.Sprintf("SELECT %s, 1 - (%s <=> $1) as similarity_score, %s FROM %s ORDER BY similarity_score DESC LIMIT %d", store.searchKey, store.embeddingStoreColumnName, strings.Join(store.queryAttributes, ","), store.tableName, numDocuments)

	fmt.Println("search query -", sqlQuery)

	return sqlQuery
}

func (store PgVectorStore) _generateSelectQuery(numDocuments int, threshold float64) string {

	var attribsToQuery []string

	//attribsToQuery = append(attribsToQuery, store.searchKey)
	attribsToQuery = append(attribsToQuery, store.queryAttributes...)

	sqlQuery := fmt.Sprintf("SELECT %s, 1 - (%s <=> $1) as similarity FROM %s ORDER BY similarity DESC LIMIT %d", strings.Join(attribsToQuery, ","), store.embeddingColumnName, store.tableName, numDocuments)

	if threshold > 0 {
		sqlQuery = sqlQuery + " WHERE similarity > " + strconv.FormatFloat(threshold, 'E', -1, 64)
	}

	return sqlQuery
}

func convertVector(v []float64) []float32 {
	v32 := make([]float32, len(v))
	for i, f := range v {
		v32[i] = float32(f)
	}
	return v32
}
