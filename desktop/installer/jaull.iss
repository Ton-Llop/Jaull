; Jaull desktop installer (Inno Setup 6). Built by build.ps1, which prepares the staging folder:
; the app and a portable Python with Jaull and all its dependencies, as uv.lock pins them.
; Per-user install: no administrator rights.

#ifndef Stage
  #define Stage "..\target\installer\stage"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{5B2E0C1F-3A7E-4E7B-9C61-1F2B7D9A4C21}
AppName=Jaull
AppVersion={#AppVersion}
AppPublisher=Antoni Llop
AppPublisherURL=https://github.com/Ton-Llop/Jaull
DefaultDirName={localappdata}\Programs\Jaull
DefaultGroupName=Jaull
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\target\installer
OutputBaseFilename=Jaull-Setup-{#AppVersion}
SetupIconFile=..\icon.ico
UninstallDisplayIcon={app}\Jaull.exe
LicenseFile=..\..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#Stage}\Jaull.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Stage}\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Jaull"; Filename: "{app}\Jaull.exe"
Name: "{group}\Uninstall Jaull"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Jaull"; Filename: "{app}\Jaull.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Jaull.exe"; Description: "Open Jaull"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Python writes bytecode caches at run time; they are not in the file list above.
Type: filesandordirs; Name: "{app}\python"
