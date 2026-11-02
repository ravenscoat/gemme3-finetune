$ErrorActionPreference = 'Stop'
$request = @{
    model = 'gemma3:4b'
    prompt = 'Reply with exactly one word: READY'
    stream = $false
    keep_alive = '5m'
    options = @{
        num_ctx = 2048
        num_predict = 16
        temperature = 0
    }
} | ConvertTo-Json -Depth 5
$result = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/generate' -Method Post -ContentType 'application/json' -Body $request -TimeoutSec 180
if (-not $result.done -or [string]::IsNullOrWhiteSpace($result.response)) {
    throw 'Ollama did not return a completed nonempty response.'
}
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'gemma_smoke_test.json') -Encoding utf8
Write-Output "Model: $($result.model)"
Write-Output "Response: $($result.response.Trim())"
if ($result.eval_duration -gt 0) {
    Write-Output ('Generation tokens/sec: {0:N1}' -f ($result.eval_count * 1e9 / $result.eval_duration))
}
ollama ps
