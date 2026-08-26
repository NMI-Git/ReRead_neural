# Results tables

Generated from the committed corpora and models.

| Table | Regenerate with |
|---|---|
| 1 — corpus descriptives | `python corpus_measures.py` |
| 2 — model performance | `python two_deck.py --corpus <KEY> --mode <N>` for N = 1–7 |

Table 2 maps to the modes as: Real Words = 1, Random String = 2, Single
Repeated Letter = 3, Double Letter Substitution = 4, Letter Transposition = 5,
`SILE`/`SLNE` = 6 sub-modes 1/2, `SILNECE`/`SILOPCE` = 7 sub-modes 1/2.

LaTeX source for both tables is in [`tables.tex`](tables.tex) and
repeated at the bottom of this file for copy-paste.

## Table 1. Descriptive information about the corpora

| Orthographic Measure | French M | SD | FIN M | SD | FIN random M | SD | t | p |
|---|---|---|---|---|---|---|---|---|
| Neighbour words # | 0.40 | 0.78 | 0.67 | 0.88 | 0.00 | 0.00 | -10.25 | <.001 |
| Levenshtein distance 20, mean | 3.12 | 0.48 | 3.05 | 0.45 | 4.66 | 0.14 | 5.11 | <.001 |
| Levenshtein distance 20, sd | 0.54 | 0.16 | 0.59 | 0.17 | 0.49 | 0.09 | -10.99 | <.001 |
| Spread (#letters yielding a neighbour) | 0.34 | 0.59 | 0.58 | 0.74 | 0.00 | 0.00 | -11.62 | <.001 |
| Uniqueness point | 4.34 | 1.04 | 5.07 | 1.31 | 3.15 | 0.45 | -19.76 | <.001 |

N = 1985 French, 2000 in both Finnish corpora. The test compares French against the non-random Finnish corpus (Welch's *t*).

These describe the word lists, not the model, so they are unaffected by the
model changes.

**One discrepancy against the previously published Table 1.** The French and
Finnish columns reproduce the published values to two decimal places on all five
measures. Two FIN random cells do not: Levenshtein distance 20 mean is 4.66 here
against 4.47 published, and its sd 0.49 against 0.67. Every other FIN random
cell matches exactly. Since the same code reproduces French and Finnish
precisely, the method is not the cause — either the random corpus was
regenerated after the published table was produced, or those two cells were
computed differently. Worth resolving before the values are reused.

The *t* values differ slightly from the published ones (for example −10.25 here
against −10.40) because these use Welch's test, which does not assume equal
variances. All five remain significant at *p* < .001 either way.

## Table 2. Model performance

| Test Condition | FR Accur. | # | Err. | FIN Accur. | # | Err. | FIRND Accur. | # | Err. |
|---|---|---|---|---|---|---|---|---|---|
| Real Words | 100.0% | 13895 | 0 | 100.0% | 14000 | 0 | 100.0% | 14000 | 0 |
| Random String | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 |
| Single Repeated Letter | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 | 100.0% | 1000 | 0 |
| Double Letter Substitution | 99.7% | 1985 | 5 | 99.7% | 2000 | 6 | 100.0% | 2000 | 0 |
| Letter Transposition | 87.5% | 1985 | 248 | 67.0% | 2000 | 659 | 98.5% | 2000 | 29 |

### Priming

Proportion of primes activating **their own target word's** lexical unit above 0.5. These are effects, not errors: the prediction is `SILE > SLNE` and `SILNECE > SILOPCE`.

| Condition | FR % | # | n | FIN % | # | n | FIRND % | # | n |
|---|---|---|---|---|---|---|---|---|---|
| SILE-SILENCE | 1.21% | 24 | 1985 | 0.75% | 15 | 2000 | 0.00% | 0 | 2000 |
| SLNE-SILENCE | 0.10% | 2 | 1985 | 0.20% | 4 | 2000 | 0.00% | 0 | 2000 |
| SILNECE-SILENCE | 30.43% | 604 | 1985 | 25.95% | 519 | 2000 | 18.95% | 379 | 2000 |
| SILOPCE-SILENCE | 4.94% | 98 | 1985 | 6.75% | 135 | 2000 | 0.05% | 1 | 2000 |

## Table 3. Proximity effect (hidden-layer distances, position x position)

Off-diagonal cells of the 13 x 13 matrix, 156 per corpus. Regenerate with
`python representation_stats.py`.

| Model | M | SD | n |
|---|---|---|---|
| French | 2.45 | 0.29 | 156 |
| Finnish | 2.97 | 0.36 | 156 |
| Random Finnish | 3.91 | 0.55 | 156 |

French vs Finnish: *F*(1, 310) = 203.57, *p* < .001.

## Table 4. Letter cluster effect (hidden-layer distances, letter x letter)

Full letter-by-letter matrix, diagonal included: 37 x 37 for French, 23 x 23 for
the Finnish corpora.

| Model | M | SD | n |
|---|---|---|---|
| French | 2.89 | 0.70 | 1369 |
| Finnish | 3.31 | 0.44 | 529 |
| Random Finnish | 4.16 | 0.15 | 529 |

French vs Finnish: *F*(1, 1896) = 164.66, *p* < .001.

### Training length confounds the random-model comparison

Criterion stopping halts each model at a different epoch — Finnish 421, French
404 — but the random corpus never meets the criterion and runs to the
2000-epoch ceiling. It therefore receives roughly five times the training of the
other two, and longer training increases representational separation on its own.

Retraining the random lower deck for 421 epochs, matching Finnish, removes the
difference entirely:

| Random Finnish lower deck | proximity M (SD) | letter cluster M (SD) |
|---|---|---|
| 2000 epochs (committed, criterion never met) | 3.91 (0.55) | 4.16 (0.15) |
| **421 epochs (matched to Finnish)** | **2.92 (0.48)** | **3.24 (0.16)** |
| *Finnish at 421 epochs, for reference* | *2.97 (0.36)* | *3.31 (0.44)* |

At equal training the random model is indistinguishable from Finnish, and
marginally lower rather than higher on both measures. Any claim that a random
training scheme *improves* letter position or identity coding should therefore
be attributed to training length, not to the corpus.

The tables above report the committed models as they stand; the matched-epoch
figures are a control, and the model producing them is not saved.

## LaTeX source

```latex
% Generated by scripts in this repository from the committed
% corpora and models. See results/TABLES.md for provenance.

\begin{table}[htbp]
\centering
\caption{Descriptive information about the corpora.}
\label{tab:corpora}
\begin{tabular}{|l|cc|cc|cc|c|c|}
\hline
 & \multicolumn{2}{c|}{\textbf{French}} & \multicolumn{2}{c|}{\textbf{FIN}} & \multicolumn{2}{c|}{\textbf{FIN random}} & & \\
\textbf{Orthographic Measure} & M & SD & M & SD & M & SD & t & p \\
\hline
Neighbour words \# & 0.40 & 0.78 & 0.67 & 0.88 & 0.00 & 0.00 & -10.25 & <.001 \\
Levenshtein distance 20, mean & 3.12 & 0.48 & 3.05 & 0.45 & 4.66 & 0.14 & 5.11 & <.001 \\
Levenshtein distance 20, sd & 0.54 & 0.16 & 0.59 & 0.17 & 0.49 & 0.09 & -10.99 & <.001 \\
Spread (\#letters yielding a neighbour) & 0.34 & 0.59 & 0.58 & 0.74 & 0.00 & 0.00 & -11.62 & <.001 \\
Uniqueness point & 4.34 & 1.04 & 5.07 & 1.31 & 3.15 & 0.45 & -19.76 & <.001 \\
\hline
\end{tabular}

\vspace{2mm}
\begin{minipage}{0.9\textwidth}
\footnotesize \textbf{Note:} N = 1985 in French corpus, 2000 in both Finnish corpora. The statistical test results are between the French and the non-random Finnish corpus (Welch's $t$-test).
\end{minipage}
\end{table}


\begin{table}[htbp]
\centering
\caption{Model performance.}
\label{tab:performance}
\begin{tabular}{|l|ccc|ccc|ccc|}
\hline
 & \multicolumn{3}{c|}{\textbf{French Model}} & \multicolumn{3}{c|}{\textbf{Finnish Model}} & \multicolumn{3}{c|}{\textbf{Random Finnish Model}} \\
\textbf{Corpus size} & \multicolumn{3}{c|}{1985 items} & \multicolumn{3}{c|}{2000 items} & \multicolumn{3}{c|}{2000 items} \\
\hline
\textbf{Test Condition} & Accur.\ (\%) & \# & Errors(\#) & Accur.\ (\%) & \# & Errors(\#) & Accur.\ (\%) & \# & Errors(\#) \\
\hline
Real Words & 100.0\% & 13895 & 0 & 100.0\% & 14000 & 0 & 100.0\% & 14000 & 0 \\
Random String & 100.0\% & 1000 & 0 & 100.0\% & 1000 & 0 & 100.0\% & 1000 & 0 \\
Single Repeated Letter & 100.0\% & 1000 & 0 & 100.0\% & 1000 & 0 & 100.0\% & 1000 & 0 \\
Double Letter Substitution & 99.7\% & 1985 & 5 & 99.7\% & 2000 & 6 & 100.0\% & 2000 & 0 \\
Letter Transposition & 87.5\% & 1985 & 248 & 67.0\% & 2000 & 659 & 98.5\% & 2000 & 29 \\
\hline
\multicolumn{10}{|c|}{\textbf{Relative Positioning Priming}} \\
\hline
\textbf{Condition} & Primed (\%) & \# & Primed(\#) & Primed (\%) & \# & Primed(\#) & Primed (\%) & \# & Primed(\#) \\
\hline
SILE-SILENCE & 1.21\% & 1985 & 24 & 0.75\% & 2000 & 15 & 0.00\% & 2000 & 0 \\
SLNE-SILENCE & 0.10\% & 1985 & 2 & 0.20\% & 2000 & 4 & 0.00\% & 2000 & 0 \\
\hline
\multicolumn{10}{|c|}{\textbf{Transposed Letter Priming}} \\
\hline
\textbf{Condition} & Primed (\%) & \# & Primed(\#) & Primed (\%) & \# & Primed(\#) & Primed (\%) & \# & Primed(\#) \\
\hline
SILNECE-SILENCE & 30.43\% & 1985 & 604 & 25.95\% & 2000 & 519 & 18.95\% & 2000 & 379 \\
SILOPCE-SILENCE & 4.94\% & 1985 & 98 & 6.75\% & 2000 & 135 & 0.05\% & 2000 & 1 \\
\hline
\end{tabular}

\vspace{2mm}
\begin{minipage}{0.95\textwidth}
\footnotesize \textbf{Note:} Recognition and nonword conditions report accuracy at the 0.9 criterion, where an error is a nonword whose winning lexical unit reached threshold. The priming blocks report the proportion of primes that activated \emph{their own target's} unit above 0.5, following Dandurand et al.\ (2013, \S3.2); these are effects to be measured, not errors, and the prediction is SILE $>$ SLNE and SILNECE $>$ SILOPCE.
\end{minipage}
\end{table}
```
