function DownloadAVBin () {
    $webclient = New-Object System.Net.WebClient
    $basedir = $pwd.Path + "\"
    Write-Host "Arch:" $Env:PYTHON_ARCH
    if ($Env:PYTHON_ARCH -eq "32") {
        $url = "https://staff.washington.edu/larsoner/avbin.dll"
        $filepath = "C:\Windows\SysWOW64\avbin.dll"
    } else {
        $url = "https://staff.washington.edu/larsoner/avbin64.dll"
        $filepath = "C:\Windows\system32\avbin64.dll"
    }
    Write-Host "Downloading" $url
    $retry_attempts = 2
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
        Write-Host "File saved at" $filepath
    } else {
        # Retry once to get the error message if any at the last try
        $webclient.DownloadFile($url, $filepath)
    }
}

function main () {
    DownloadAVBin
}

main
