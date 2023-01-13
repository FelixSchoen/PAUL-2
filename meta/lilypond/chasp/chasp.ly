% lilypond --pspdfopt=TeX -dcrop chasp.ly

\version "2.22.2"
\language "english"

\new GrandStaff <<
\new Staff \relative c'' {
\clef treble
\key g \major

r4
r8
b,8(
<e g>4\staccato)
r4

r4
r8
a,8(
<c g'>4\staccato)
r4

r4
r8
b8(
<d a'>4\staccato)
r4

r4
r8
b8(
<e g>4\staccato)
r4

}


\new Staff \relative c {
\clef bass
\key g \major

e4.
r8
r4
b4

e4.
r8
r4
a,4

f-sharp'4.
r8
r4
b,4

e4.
r8
r4
b4

}
>>