% lilypond --pspdfopt=TeX -dcrop name.ly

\version "2.22.2"
\language "english"

\new Staff \relative c'' {
\clef treble
\key c \major

<c e g>4
<b d g>4
<a c e>4
<a c f>4

}