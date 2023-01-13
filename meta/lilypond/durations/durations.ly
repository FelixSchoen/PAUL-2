% lilypond --pspdfopt=TeX -dcrop fantaisie_impromptu.ly

\version "2.22.2"
\language "english"

\new GrandStaff <<
\new Staff \relative c' {
\clef treble

\autoBeamOff

c1

% Bar

c2
c4
c8
c16
c32
r32

\autoBeamOn

c2
c4
c8
c16
c32
r32

% Bar
c4.
c4.
r4

}


\new Staff \relative c {
\clef bass

r1

% Bar

r2
r4
r8
r16
r32
r32

r2
r4
r8
r16
r32
r32

% Bar
r4
r8
r4
r8
r4

}
>>