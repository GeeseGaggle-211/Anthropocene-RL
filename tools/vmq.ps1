# Laptop wrapper for the VM job queue: `vmq <subcommand> [args]` runs `elderq` on the training VM over gcloud IAP SSH.
#   powershell -File Tools\vmq.ps1 status --json
#   powershell -File Tools\vmq.ps1 submit C:\path\to\job.yaml     (the file is copied up first)
# The command travels base64-encoded so no quoting survives to the remote shell. See docs/VM_QUEUE.md.
# Named parameters only, so the subcommand and its arguments (vmq status, vmq logs <job>) all reach $Rest.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Vm = 'anthropocene-trainer-60',
    [string]$Zone = 'us-central1-c',
    [string]$Project = 'anthropocene-508912',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
if (-not $Rest -or $Rest.Count -eq 0) { $Rest = @('help') }

function Quote([string]$s) { "'" + $s.Replace("'", "'\''") + "'" }

function Invoke-Remote([string]$script) {
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
    # Print the remote output here and return only the exit code (a function's output would otherwise become its value).
    gcloud compute ssh $Vm --project=$Project --zone=$Zone --tunnel-through-iap --strict-host-key-checking=yes --quiet "--command=echo $b64 | base64 -d | bash" | Out-Host
    return $LASTEXITCODE
}

$sub = $Rest[0]
$args2 = @()
if ($Rest.Count -gt 1) { $args2 = $Rest[1..($Rest.Count - 1)] }

if ($sub -eq 'submit') {
    $file = $args2 | Where-Object { $_ -notlike '--*' } | Select-Object -First 1
    if (-not $file) { Write-Error 'usage: vmq submit <local jobfile> [--json]'; exit 2 }
    $full = (Resolve-Path -LiteralPath $file).Path
    $dir = Split-Path -Parent $full
    $leaf = Split-Path -Leaf $full
    # pscp fails on absolute Windows paths: copy from the file's own directory using a relative name.
    Push-Location $dir
    try {
        $remote = "vmq-$leaf"
        gcloud compute scp $leaf "${Vm}:$remote" --project=$Project --zone=$Zone --tunnel-through-iap --strict-host-key-checking=yes --quiet
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally { Pop-Location }
    $parts = @('submit', ('"$HOME/' + $remote + '"')) + @($args2 | Where-Object { $_ -like '--*' } | ForEach-Object { Quote $_ })
    $cmd = '$HOME/queue/bin/elderq ' + ($parts -join ' ')
}
else {
    $cmd = '$HOME/queue/bin/elderq ' + ((@($sub) + $args2 | ForEach-Object { Quote $_ }) -join ' ')
}
exit (Invoke-Remote $cmd)
