#!/bin/bash
# render.sh DOCX PDF : export via Pages
osascript <<OSA
tell application "Pages"
  set d to open POSIX file "$1"
  delay 4
  export d to POSIX file "$2" as PDF
  close d saving no
end tell
OSA
