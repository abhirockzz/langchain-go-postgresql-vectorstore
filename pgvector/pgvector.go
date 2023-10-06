package pgvector

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
	pgv "github.com/pgvector/pgvector-go"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

type Store struct {
	embedder embeddings.Embedder
	pool     *pgxpool.Pool

	// name of the table
	tableName string

	//name of the column in which text data will be stored. this will come from the "PageContent" of the langchain doc
	textColumnName string

	//name of the column in which embedded vector data will be stored. the data from "PageContent" of the langchain doc will go through the embedding (vector creation) process
	embeddingStoreColumnName string

	//if true, the langchain doc Metadata will be saved to postgresql as well. in that case the, column(s) needs to exist in advance
	saveMetadata bool

	// attributes for similarity search
	//searchKey       string   // name of the column whose value needs to returned by search

	//optional - data for these columns will be added to resulting langchain doc Metadata
	QueryAttributes []string
}

func New(pgConnectionString, tableName, embeddingStoreColumnName, textColumnName string, saveMetadata bool, embedder embeddings.Embedder) (Store, error) {
	//connection string example - postgres://postgres:postgres@localhost/postgres
	pool, err := pgxpool.New(context.Background(), pgConnectionString)

	if err != nil {
		return Store{}, err
	}

	return Store{embedder: embedder,
		tableName:                tableName,
		embeddingStoreColumnName: embeddingStoreColumnName,
		textColumnName:           textColumnName,
		pool:                     pool,
		saveMetadata:             saveMetadata}, nil
}

var ErrEmbedderWrongNumberVectors = errors.New(
	"number of vectors from embedder does not match number of documents",
)

func (store Store) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) error {

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
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

		data := map[string]any{}
		data[store.embeddingStoreColumnName] = pgv.NewVector(vectors[i])
		data[store.textColumnName] = doc.PageContent

		metadata := metadatas[i]

		query, values := store.generateInsertQueryWithValues(data, metadata)
		//fmt.Println("generated query:", query)

		_, err := store.pool.Exec(context.Background(), query, values...)
		if err != nil {
			return err
		}
	}

	return nil
}

func (store Store) generateInsertQueryWithValues(data, metadata map[string]any) (string, []any) {

	//INSERT INTO test_table (data, embedding) VALUES ($1, $2)
	//INSERT INTO test_table (data, embedding, other_data) VALUES ($1, $2, $3)

	// generate column names and placeholders dynamically
	var columns []string
	var placeholders []string
	var values []any

	for column, value := range data {
		columns = append(columns, column)
		placeholders = append(placeholders, fmt.Sprintf("$%d", len(placeholders)+1))
		values = append(values, value)
	}

	if store.saveMetadata {
		for column, value := range metadata {
			columns = append(columns, column)
			placeholders = append(placeholders, fmt.Sprintf("$%d", len(placeholders)+1))
			values = append(values, value)
		}
	}

	sqlQuery := fmt.Sprintf(
		"INSERT INTO %s (%s) VALUES (%s)",
		store.tableName,
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", "),
	)

	return sqlQuery, values
}

func (store Store) SimilaritySearch(ctx context.Context, searchString string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {

	//fmt.Println("similarity search for", searchString, "with max docs", numDocuments)

	vector, err := store.embedder.EmbedQuery(ctx, searchString)
	if err != nil {
		return nil, err
	}

	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}

	query := store.generateSelectQuery(numDocuments, opts.ScoreThreshold)

	rows, err := store.pool.Query(context.Background(), query, pgv.NewVector(vector))

	if err != nil {
		return nil, err
	}

	defer rows.Close()

	docs := []schema.Document{}
	doc := schema.Document{}

	for rows.Next() {
		vals, err := rows.Values()

		if err != nil {
			return nil, err
		}

		doc.PageContent = vals[0].(string)

		score := vals[1].(float64)
		doc.Score = float32(score)

		metadata := make(map[string]any)
		for i := 2; i <= len(vals)-1; i++ {
			metadata[store.QueryAttributes[i-2]] = vals[i]
		}

		doc.Metadata = metadata

		docs = append(docs, doc)
	}

	return docs, nil
}

const queryFormatWithQueryAttributes = "SELECT %s, 1 - (%s <=> $1) as similarity_score, %s FROM %s WHERE 1 - (embedding <=> $1) > %v ORDER BY similarity_score DESC LIMIT %d"

const queryFormat = "SELECT %s, 1 - (%s <=> $1) as similarity_score FROM %s WHERE 1 - (embedding <=> $1) > %v ORDER BY similarity_score DESC LIMIT %d"

func (store Store) generateSelectQuery(numDocuments int, threshold float32) string {

	//SELECT data, 1 - (embedding <=> $1) as similarity_score FROM test_table WHERE 1 - (embedding <=> $1) > 0.5 ORDER BY similarity_score DESC LIMIT 5
	//SELECT data, 1 - (embedding <=> $1) as similarity_score, other_data FROM test_table WHERE 1 - (embedding <=> $1) > 0.5 ORDER BY similarity_score DESC LIMIT 5

	var sqlQuery string

	if len(store.QueryAttributes) > 0 {

		sqlQuery = fmt.Sprintf(queryFormatWithQueryAttributes, store.textColumnName, store.embeddingStoreColumnName, strings.Join(store.QueryAttributes, ","), store.tableName, threshold, numDocuments)

	} else {
		sqlQuery = fmt.Sprintf(queryFormat, store.textColumnName, store.embeddingStoreColumnName, store.tableName, threshold, numDocuments)
	}

	//fmt.Println("search query -", sqlQuery)

	return sqlQuery
}
