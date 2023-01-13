% lilypond --pspdfopt=TeX -dcrop fantaisie_impromptu.ly

\version "2.22.2"
\language "english"

\new GrandStaff <<
\new Staff \relative c'' {
\clef treble
\key cs \minor

\override Score.BarNumber.break-visibility = ##(#f #t #t)
\set Score.currentBarNumber = #5
% Permit first bar number to be printed
\bar ""

r16
gs16(
\< % Crescendo
a16
gs16
fss16
gs16
cs16
e16
ds16
cs16
ds16
cs16
bs16
cs16
e16
gs16)
\!

% Bar

r16
gs,16(
a16
gs16
fss16
\<
gs16
cs16
e16
ds16
cs16
ds16
cs16
bs16
cs16
e16
gs16)
\!

}


\new Staff \relative c {
\clef bass
\key cs \minor

\tuplet 6/4 {
cs8->[ % Accent, Group Notes
gs'8
cs8
e8->
cs8
gs8]
}

\tuplet 6/4 {
e8->[
gs8
cs8
gs'8->
cs,8
gs8]
}

% Bar

\tuplet 6/4 {
cs,8->[ % Accent, Group Notes
gs'8
cs8
e8->
cs8
gs8]
}

\tuplet 6/4 {
e8->[
gs8
cs8
gs'8->
cs,8
gs8]
}

}
>>