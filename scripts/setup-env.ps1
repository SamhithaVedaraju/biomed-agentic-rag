# Creates .env in the project root. The API key is read as hidden input and never echoed.
$root = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $root ".env"

$secure = Read-Host "Anthropic API key (hidden; press Enter to skip and use free local Ollama)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim()
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
$backend = if ($key) { "claude" } else { "ollama" }

if ($key -and -not $key.StartsWith("sk-ant-")) {
    Write-Host "That doesn't look like an Anthropic key (it should start with sk-ant-). Nothing saved." -ForegroundColor Red
    exit 1
}

$email = Read-Host "Your email for NCBI/PubMed requests"

$lines = Get-Content (Join-Path $root ".env.example") | ForEach-Object {
    if ($_ -like "ANTHROPIC_API_KEY=*") { "ANTHROPIC_API_KEY=$key" }
    elseif ($_ -like "BIORAG_LLM_BACKEND=*") { "BIORAG_LLM_BACKEND=$backend" }
    elseif ($_ -like "NCBI_EMAIL=*") { "NCBI_EMAIL=$email" }
    else { $_ }
}
Set-Content -Path $envPath -Value $lines -Encoding ascii
Remove-Variable key, secure

Write-Host "Saved $envPath. You can close this tab and tell Claude 'done'." -ForegroundColor Green
