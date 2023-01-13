% lilypond --pspdfopt=TeX -dcrop file.ly

\version "2.22.2"
\language "english"
\layout {
  indent = 0.0
}

\new GrandStaff <<
\new Staff \relative c'' {
\clef treble
\key f \major

\autoBeamOff

r16
<< f8.~ c'8.~ f8.~ >>[
<< f,16 c'16 f16 >>]
<< e,8.~ c'8.~ e8.~ >>[
<< e,16 c'16 e16 >>]
g,8.~
g16[
r16
<< f16 g'16] >>
r16

r16
<< f,8.~ e'8.~ >>
<< f,16 e'16[ >>
<< e,8.~ c'8.~] >>
<< e,16 c'16[ >>
<< e,8.~ c'8.] >>
<< e,16 d'16~[ >>
d16
<< e,8 c'8] >>
<< e,16 f16~ bf16~ c16 f16~[ >>
<< f,8. bf8.~ f'8.] >>
<< f,4 bf4 >>
f8~[
\tuplet 3/2 {f8~ << f16~ g16~] >>}
<< f8~ g8~[ >>
\tuplet 3/2 {<< f8~ g8 >> f16]}

<< c4 e4 g4 e'4 >>
<< c,8~ e8~ g8~[ >>
<< c,8 e8 g8] >>
\tuplet 3/2 {g8~[ << f16~ g16 >>}
f8]
\tuplet 3/2 {f4 r8}

}


\new Staff \relative c {
\clef bass
\key f \major

r8
f'16
r16
<< e16 f16[ >>
r16
ef8]
<< e16 f16~[ >>
f16
<< c16~ f16 >>
c16~]
<< c16~ d16~ f16[ >>
<< c16 d16 >>
<< e8 f8] >>

<< e8 f8[ >>
f8~]
f16[
r16
f16]
r16
f16[
r16
\tuplet 3/2 {r16 f8]}
\tuplet 3/2 {r8 f8 r8}

e8[
e8~]
\tuplet 3/2 {<< e8~ f8[ >> e16}
\tuplet 3/2 {<< d8 f8~ >> f16]}
\tuplet 3/2 {c8[ r16}
f16
<< c16 f16~] >>
f16[
<< c16~ d16 f16~ >>
<< g,16 a16 c16~ f16~ >>
<< c16 d16 f16] >>

<< e16~[ f16 >>
<< e8. d8.~] >>
d16[
r16
\tuplet 3/2 {r8 a16~]}

\tuplet 3/2 {
a16[
r16
<< d,8~ a'8 >>
d,16
<< d16~ a'16~] >>
}
\tuplet 3/2 {
<< d,8~ a'8~[ >>
<< d,8 a'8~[ >>
a16
d,16]
}

\bar "|."

}
>>