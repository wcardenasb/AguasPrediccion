
#define AppName "Aguas y Aguas - Predicción de consumo"
#define AppVersion "1.0.1"
#define AppExe "AguasPrediccion.exe"

[Setup]
AppId={{D04F238A-6B2F-445A-9D82-B43D987A5612}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Aguas y Aguas
DefaultDirName={localappdata}\Programs\AguasPrediccion
DefaultGroupName=Aguas y Aguas
OutputDir=instaladores
OutputBaseFilename=Instalador_AguasPrediccion_1.0.1
SetupIconFile=gota_aguas.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear acceso directo en el escritorio"; Flags: checkedonce

[Files]
Source: "dist\AguasPrediccion\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "gota_aguas.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Aguas y Aguas"; Filename: "{app}\AguasPrediccion.exe"; IconFilename: "{app}\gota_aguas.ico"
Name: "{autodesktop}\Aguas y Aguas"; Filename: "{app}\AguasPrediccion.exe"; IconFilename: "{app}\gota_aguas.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Abrir Aguas y Aguas"; Flags: nowait postinstall skipifsilent
