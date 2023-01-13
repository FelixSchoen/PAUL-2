% lilypond --pspdfopt=TeX -dcrop fantaisie_impromptu.ly

\version "2.22.2"
\language "english"

\new Staff {
\clef treble

c8-"C3"
d8-"D3"
e8-"E3"
f8-"F3"
g8-"G3"
a8-"A3"
b8-"B3"
c'8-"C4"

d'8-"D4"
e'8-"E4"
f'8-"F4"
g'8-"G4"
a'8-"A4"
b'8-"B4"
c''8-"C5"
r8

}


\new Staff {
\clef bass

c8-"C3"
d8-"D3"
e8-"E3"
f8-"F3"
g8-"G3"
a8-"A3"
b8-"B3"
c'8-"C4"

d'8-"D4"
e'8-"E4"
f'8-"F4"
g'8-"G4"
a'8-"A4"
b'8-"B4"
c''8-"C5"
r8

}