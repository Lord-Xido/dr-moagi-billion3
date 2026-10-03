param([string]$Path=(Join-Path (Split-Path $PSScriptRoot -Parent) 'dm3d_1GB.vmem'))
$ErrorActionPreference='Stop'
$Path=[System.IO.Path]::GetFullPath($Path)
# CreateNew rejects existing substrates rather than replacing them.
$f=[System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew)
$f.Dispose()
fsutil sparse setflag $Path
if($LASTEXITCODE -ne 0){throw 'Sparse flag failed; empty file retained for inspection'}
$f=[System.IO.File]::Open($Path,[System.IO.FileMode]::Open)
$w=[System.IO.BinaryWriter]::new($f)
try {
 $f.SetLength(1000000000)
 $w.Write([System.Text.Encoding]::ASCII.GetBytes('DM3D0001'))
 $w.Write([uint64]1)
 $w.Write([uint64]999000000)
 $w.Write([uint64]0)
 $f.Position=65536
 [uint64]$seed=1
 for($i=0;$i -lt 262144;$i++){
  $seed=($seed*1664525+1013904223) -band 4294967295
  $w.Write([single](($seed -shr 16)/65536.0))
 }
 $w.Flush()
} finally {$w.Dispose()}
fsutil sparse queryflag $Path
"Created $Path with 1 GB logical length and one deterministic seed tile."
