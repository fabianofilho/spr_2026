Param(
    [string]$RemoteUrl,
    [string]$UserName,
    [string]$UserEmail
)

if (-not $RemoteUrl) {
    Write-Host "Usage: .\switch_git_remote.ps1 -RemoteUrl <git_url> [-UserName <name>] [-UserEmail <email>]"
    exit 1
}

git remote set-url origin $RemoteUrl
Write-Host "Set origin -> $RemoteUrl"

if ($UserName) {
    git config user.name "$UserName"
    Write-Host "Set git user.name -> $UserName"
}

if ($UserEmail) {
    git config user.email "$UserEmail"
    Write-Host "Set git user.email -> $UserEmail"
}

Write-Host "Done. Run 'git remote -v' to confirm and then 'git push origin main'"
