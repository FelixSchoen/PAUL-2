% lilypond --pspdfopt=TeX -dcrop fantaisie_impromptu.ly

\version "2.22.2"
\language "english"

\new GrandStaff <<
\new Staff \relative c' {
\clef treble

\autoBeamOff

c8
c8
c8
c8
c4
c4

}

\new Staff \relative c {
\clef bass


\tuplet 3/2 {c8 8 8}

\tuplet 5/4 {c16 16 16 16 16}

\tuplet 3/2 {c4 4 4}

}
>>