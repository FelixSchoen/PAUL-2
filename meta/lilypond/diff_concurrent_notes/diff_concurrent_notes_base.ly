% lilypond --pspdfopt=TeX -dcrop name.ly

\version "2.22.2"
\language "english"

\new Staff \relative c'' {
\clef treble
\key c \major

c4
b4
a4
f4

}