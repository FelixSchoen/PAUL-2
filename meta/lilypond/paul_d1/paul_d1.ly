% lilypond --pspdfopt=TeX -dcrop file.ly

\version "2.22.2"
\language "english"

\new GrandStaff <<
\new Staff \relative c'' {
\clef treble
\key ef \major

r4
g4
<< af4 c4 >>
g4

<< f4~ g4~ >>
<< f4 g4 af4 >>
<< f4 af4~ f'4 >>
af,4

r4
g4~
<< g4~ af c >>
bf

<< g4 bf4 >>
<< g4 ef'4 >>
bf2

}


\new Staff \relative c {
\clef bass
\key ef \major

r2
r4
r8
c'8

c4
ef4
r2

ef4
f4
gf4
af4

bf4
r4
r2

\bar "|."

}
>>