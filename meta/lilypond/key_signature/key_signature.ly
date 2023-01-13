% lilypond --pspdfopt=TeX -dcrop fantaisie_impromptu.ly

\version "2.22.2"
\language "english"

\new Staff \relative c'' {
\clef treble
\key c \major

cs,4
ds4
es4
fs4
gs4
as4
bs4
cs4

\key cs \major

cs,4
ds4
es4
fs4
gs4
as4
bs4
cs4

}