% lilypond --pspdfopt=TeX -dcrop name.ly

\version "2.22.2"
\language "english"

\new Staff \relative c'' {
\clef treble
\key e \major

\time 2/2

\autoBeamOff

\tuplet 3/2 {gs,8[ e'8 cs8]}

\tuplet 3/2 {e8[ gs,8 cs8]}

\tuplet 3/2 {cs8[ gs8 e'8]}

\tuplet 3/2 {e8[ cs8 gs8]}

}