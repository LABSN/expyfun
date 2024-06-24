function DownloadDLL($dll) {
    $url = "https://staff.washington.edu/larsoner/" + $dll
    $filepath = "C:\Windows\system32\" + $dll
    $webclient = New-Object System.Net.WebClient
    $basedir = $pwd.Path + "\"
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
    DownloadDLL("avcodec-58.dll")
    DownloadDLL("avformat-58.dll")
    DownloadDLL("avutil-56.dll")
    DownloadDLL("swresample-3.dll")
    DownloadDLL("swscale-5.dll")
}

main
