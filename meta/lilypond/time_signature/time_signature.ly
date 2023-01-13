% lilypond --pspdfopt=TeX -dcrop fantaisie_impromptu.ly

\version "2.22.2"
\language "english"

\new Staff \relative c' {
\clef treble

\time 4/4

c4
c4
c4
c4

\time 3/4

c4
c4
c4

\time 2/4

c4
c4

\time 6/8

c4
c4
c4

\time 9/8

c4
c4
c4
c4
c8

}
