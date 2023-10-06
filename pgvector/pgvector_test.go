package pgvector

import (
	"context"
	"testing"

	pgv "github.com/pgvector/pgvector-go"
	"github.com/stretchr/testify/assert"
	testcontainers "github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/tmc/langchaingo/schema"
)

func TestAddDocuments(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"

	postgresContainer, err := startPostgresContainer()
	assert.Nil(t, err)

	ctx := context.Background()

	defer postgresContainer.Terminate(ctx)

	ip, err := postgresContainer.Host(ctx)
	assert.Nil(t, err)

	port, err := postgresContainer.MappedPort(ctx, "5432")
	assert.Nil(t, err)

	pgConnString := "postgres://postgres:postgres@" + ip + ":" + port.Port() + "/postgres"

	saveDocMetadtaToTable := false

	pgStore, err := New(pgConnString, tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	//create extension and table
	_, err = pgStore.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector;")
	assert.Nil(t, err)

	_, err = pgStore.pool.Exec(ctx, "CREATE TABLE IF NOT EXISTS test_table (id bigserial primary key, data text, embedding vector(1));")
	assert.Nil(t, err)

	docs := []schema.Document{{PageContent: "foo"}}

	err = pgStore.AddDocuments(context.Background(), docs)
	assert.Nil(t, err)

	rows, err := pgStore.pool.Query(context.Background(), "select count(*) from test_table;")
	assert.Nil(t, err)

	for rows.Next() {
		var count int
		err = rows.Scan(&count)

		assert.Nil(t, err)
		assert.Equal(t, 1, count)
	}

	rows, err = pgStore.pool.Query(context.Background(), "select data, embedding from test_table;")
	assert.Nil(t, err)

	for rows.Next() {

		var data string
		var embedding pgv.Vector

		err = rows.Scan(&data, &embedding)

		assert.Nil(t, err)
		assert.Equal(t, "foo", data)
		assert.Equal(t, []float32{42.42}, embedding.Slice())
	}
}

func TestAddDocumentsWithMetadata(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"

	postgresContainer, err := startPostgresContainer()
	assert.Nil(t, err)

	ctx := context.Background()

	defer postgresContainer.Terminate(ctx)

	ip, err := postgresContainer.Host(ctx)
	assert.Nil(t, err)

	port, err := postgresContainer.MappedPort(ctx, "5432")
	assert.Nil(t, err)

	pgConnString := "postgres://postgres:postgres@" + ip + ":" + port.Port() + "/postgres"

	saveDocMetadtaToTable := true

	pgStore, err := New(pgConnString, tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	//create extension and table
	_, err = pgStore.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector;")
	assert.Nil(t, err)

	_, err = pgStore.pool.Exec(ctx, "CREATE TABLE IF NOT EXISTS test_table (id bigserial primary key, data text, other_data text, embedding vector(1));")

	assert.Nil(t, err)

	docs := []schema.Document{{PageContent: "foo", Metadata: map[string]any{"other_data": "bar"}}}

	err = pgStore.AddDocuments(context.Background(), docs)
	assert.Nil(t, err)

	rows, err := pgStore.pool.Query(context.Background(), "select count(*) from test_table;")
	assert.Nil(t, err)

	for rows.Next() {
		var count int
		err = rows.Scan(&count)

		assert.Nil(t, err)
		assert.Equal(t, 1, count)
	}

	rows, err = pgStore.pool.Query(context.Background(), "select data, embedding, other_data from test_table;")
	assert.Nil(t, err)

	for rows.Next() {

		var data string
		var embedding pgv.Vector
		var metadata string

		err = rows.Scan(&data, &embedding, &metadata)

		assert.Nil(t, err)
		assert.Equal(t, "foo", data)
		assert.Equal(t, []float32{42.42}, embedding.Slice())
		assert.Equal(t, "bar", metadata)

	}
}

func TestSimilaritySearch(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"

	postgresContainer, err := startPostgresContainer()
	assert.Nil(t, err)

	ctx := context.Background()

	defer postgresContainer.Terminate(ctx)

	ip, err := postgresContainer.Host(ctx)
	assert.Nil(t, err)

	port, err := postgresContainer.MappedPort(ctx, "5432")
	assert.Nil(t, err)

	pgConnString := "postgres://postgres:postgres@" + ip + ":" + port.Port() + "/postgres"

	saveDocMetadtaToTable := false

	pgStore, err := New(pgConnString, tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	//create extension and table
	_, err = pgStore.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector;")
	assert.Nil(t, err)

	_, err = pgStore.pool.Exec(ctx, "CREATE TABLE IF NOT EXISTS test_table (id bigserial primary key, data text, embedding vector(1));")
	assert.Nil(t, err)

	docs := []schema.Document{{PageContent: "foo"}}

	err = pgStore.AddDocuments(context.Background(), docs)
	assert.Nil(t, err)

	searchResults, err := pgStore.SimilaritySearch(ctx, "doesn't really matter", 1)
	assert.Nil(t, err)

	assert.Equal(t, 1, len(searchResults))
	assert.Equal(t, "foo", searchResults[0].PageContent)
	assert.Equal(t, float32(1), searchResults[0].Score)

}

func TestGenerateSelectQuery(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"
	saveDocMetadtaToTable := false

	pgStore, err := New("postgres://postgres:postgres@localhost/postgres", tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	//pgStore.QueryAttributes = []string{}

	query := pgStore.generateSelectQuery(5, 0.5)
	expectedQuery := "SELECT data, 1 - (embedding <=> $1) as similarity_score FROM test_table WHERE 1 - (embedding <=> $1) > 0.5 ORDER BY similarity_score DESC LIMIT 5"

	assert.Equal(t, expectedQuery, query)
}

func TestGenerateSelectQueryWithQueryAttributes(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"
	saveDocMetadtaToTable := false

	pgStore, err := New("postgres://postgres:postgres@localhost/postgres", tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	pgStore.QueryAttributes = []string{"other_data"}

	query := pgStore.generateSelectQuery(5, 0.5)
	expectedQuery := "SELECT data, 1 - (embedding <=> $1) as similarity_score, other_data FROM test_table WHERE 1 - (embedding <=> $1) > 0.5 ORDER BY similarity_score DESC LIMIT 5"

	assert.Equal(t, expectedQuery, query)
}

func TestGenerateInsertQueryWithValues(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"
	saveDocMetadtaToTable := false

	pgStore, err := New("postgres://postgres:postgres@localhost/postgres", tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	pgStore.saveMetadata = false

	//pgStore.QueryAttributes = []string{"other_data"}

	query, values := pgStore.generateInsertQueryWithValues(map[string]any{"data": "foo", "embedding": pgv.NewVector([]float32{42})}, nil)

	expectedQuery := "INSERT INTO test_table (data, embedding) VALUES ($1, $2)"
	assert.Equal(t, expectedQuery, query)

	expectedValues := []any{"foo", pgv.NewVector([]float32{42})}
	assert.Equal(t, expectedValues, values)
}

func TestGenerateInsertQueryWithValuesAndMetadata(t *testing.T) {

	tableName := "test_table"
	textColumnName := "data"
	embeddingStoreColumnName := "embedding"
	saveDocMetadtaToTable := false

	pgStore, err := New("postgres://postgres:postgres@localhost/postgres", tableName, embeddingStoreColumnName, textColumnName, saveDocMetadtaToTable, MockEmbedder{})
	assert.Nil(t, err)

	pgStore.saveMetadata = true

	//pgStore.QueryAttributes = []string{"other_data"}

	query, values := pgStore.generateInsertQueryWithValues(map[string]any{"data": "foo", "embedding": pgv.NewVector([]float32{42})}, map[string]any{"other_data": "bar"})

	expectedQuery := "INSERT INTO test_table (data, embedding, other_data) VALUES ($1, $2, $3)"
	assert.Equal(t, expectedQuery, query)

	expectedValues := []any{"foo", pgv.NewVector([]float32{42}), "bar"}
	assert.Equal(t, expectedValues, values)

}

type MockEmbedder struct{}

func (m MockEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	return [][]float32{{42.42}}, nil
}

func (m MockEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return []float32{42.42}, nil
}

func startPostgresContainer() (testcontainers.Container, error) {
	ctx := context.Background()

	req := testcontainers.ContainerRequest{
		Image:        "ankane/pgvector:latest",
		ExposedPorts: []string{"5432/tcp"},
		WaitingFor:   wait.ForListeningPort("5432/tcp"),
		Env: map[string]string{
			"POSTGRES_PASSWORD": "postgres",
			"POSTGRES_USER":     "postgres",
		},
	}
	postgresContainer, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		return nil, err
	}

	return postgresContainer, nil
}
