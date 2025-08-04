# Contributing to LLM Infrastructure

## Adding Middleware

1. Create `middleware_yourname.go`
2. Follow the pattern in `middleware_rate_limiter.go` (simplest example)
3. Add tests in `middleware_yourname_test.go`

Example structure:
```go
type yourMiddleware struct {
    next CoreLLM
    // your fields
}

func YourMiddleware(params) Middleware {
    return func(next CoreLLM) CoreLLM {
        return &yourMiddleware{next: next}
    }
}

func (m *yourMiddleware) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
    // your logic before
    response, tokensIn, tokensOut, err := m.next.DoRequest(ctx, prompt, opts)
    // your logic after
    return response, tokensIn, tokensOut, err
}

func (m *yourMiddleware) GetModel() string { return m.next.GetModel() }
func (m *yourMiddleware) SetModel(model string) { m.next.SetModel(model) }
```

## Adding Providers

1. Create `provider_yourname.go`
2. Follow the pattern in `provider_openai.go`
3. Register in init() function
4. Add tests in `provider_yourname_test.go`

Example structure:
```go
func init() {
    RegisterProviderFactory("yourprovider", newYourProvider)
}

type yourProvider struct {
    apiKey string
    model  string
    // your fields
}

func newYourProvider(config ClientConfig) (CoreLLM, error) {
    // validate config
    return &yourProvider{
        apiKey: config.APIKey,
        model:  config.Model,
    }, nil
}

func (p *yourProvider) DoRequest(ctx context.Context, prompt string, opts map[string]any) (string, int, int, error) {
    // implement API call
}

func (p *yourProvider) GetModel() string { return p.model }
func (p *yourProvider) SetModel(m string) { p.model = m }
```

## Testing

Run tests with:
```bash
go test ./infrastructure/llm/... -v
```
