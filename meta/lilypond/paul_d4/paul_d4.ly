% lilypond --pspdfopt=TeX -dcrop file.ly

\version "2.22.2"
\language "english"

\new GrandStaff <<
\new Staff \relative c'' {
\clef treble
\key e \major

r4
<< cs4 g'4 >>
ds8[
e8]
r8
bs8

<< bs4 b4~ >>
<< b4 ds4 >>
b4
<< gs4 b4 gs'4 >>

<< gs4~ gs,4 >>
gs'4
<< g,4 bs4 ds4 g4 >>
r4

<< b,4 ds4 b'4 >>
<< b,4 b'4 >>
r2


}


\new Staff \relative c {
\clef bass
\key e \major

r8
b8
<< fs'4 b4 ds4 >>
r2

<< b4 ds4 >>
e4
fs4~
<< e4 fs4 gs4 >>

<< e4 fs4 >>
r4
r4
e8~
<< ds8 e8 fs8 >>

<< ds8~ e8 fs8~ >>
<< e16~ fs16 >>
e16
e4
<< ds4 fs4~ >>
fs4

\bar "|."

}
>>