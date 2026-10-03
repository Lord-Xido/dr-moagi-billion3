$ErrorActionPreference='Stop'
$projectPath=Split-Path $PSScriptRoot -Parent
$f=[System.IO.File]::OpenRead((Join-Path $projectPath 'dm3d_1GB.vmem'))
$r=[System.IO.BinaryReader]::new($f)
try {
 if($f.Length -ne 1000000000){throw 'Length mismatch'}
 if([System.Text.Encoding]::ASCII.GetString($r.ReadBytes(8)) -ne 'DM3D0001'){throw 'Magic mismatch'}
 if($r.ReadUInt64() -ne 1 -or $r.ReadUInt64() -ne 999000000 -or $r.ReadUInt64() -ne 0){throw 'Seed index mismatch'}
 $f.Position=65536
 [uint64]$seed=1
 for($i=0;$i -lt 262144;$i++){
  $seed=($seed*1664525+1013904223) -band 4294967295
  $expected=[single](($seed -shr 16)/65536.0)
  if($r.ReadSingle() -ne $expected){throw "Seed mismatch at $i"}
 }
 $f.Position=65536+1048576
 foreach($b in $r.ReadBytes(4096)){if($b -ne 0){throw 'Unpopulated tile is not zero'}}
 'PASS backing length, header, index, all 262144 seed voxels, unpopulated zero tile'
 'C++ compilation and runtime verification NOT RUN: no Windows C++ toolchain found.'
} finally {$r.Dispose()}


