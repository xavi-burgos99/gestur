# Project licensing review

Reviewed on 2026-09-27. This document is advice for the maintainer and **does not
grant a new license**. No project-wide `LICENSE` has been activated by this cleanup.
Third-party licenses and rights previously granted remain in force.

## Requested policy

The intended policy is free use, modification, and sharing for noncommercial
purposes, with attribution to **Xavier Burgos**. Commercial use should require
his prior, explicit written permission. Source should remain available for
inspection and adaptation.

This is **source-available**, not open source as defined by the Open Source
Initiative: the [Open Source Definition](https://opensource.org/osd), sections 1
and 6, does not allow restricting commercial redistribution or business use.
MIT, Apache, GPL, and AGPL therefore cannot meet the commercial restriction.
Copyleft can require source availability in relevant circumstances; it cannot
reserve all commercial use to the original author.

## Recommendation

There are two viable routes, with a material difference in attribution:

| Route | Suitable when | Limitation |
| --- | --- | --- |
| **PolyForm Noncommercial 1.0.0**, unchanged, with a `Required Notice` | A standardized software license and preserved author notices are enough. | It does not require a visible credit in every running installation, and its institutional-use permission is broader than a blanket ban on monetized use. |
| **A custom Gestur noncommercial license**, reviewed before adoption | Every public installation must visibly credit the developer, and all commercial or promotional exploitation needs permission. | It is not a standard open-source license; precise scope, exceptions, and enforceability need legal review. |

For the strict reading of the maintainer's request, the second route is the
closer fit. A concrete [draft](gestur-license-draft.md) is provided for review,
including separate commercial permission and an explicit third-party carve-out.
Do not identify that draft as “PolyForm” or add conditions while claiming the
result is the unmodified PolyForm license.

[PolyForm Noncommercial](https://polyformproject.org/licenses/noncommercial/1.0.0)
is specifically designed for software. Its notices clause preserves a supplied
`Required Notice` when copies are distributed; running an installation is not
itself an obligation to show the author's name on screen. Its noncommercial
organizations clause also permits use by specified charitable, educational,
research, health, environmental, and government institutions regardless of
funding sources or resulting obligations. An operator's nonprofit status and a
particular use's commercial character are not interchangeable under this text.

If that standard route is selected, the proposed notice is:

```text
Required Notice: Gestur — developed by Xavier Burgos.
Required Notice: https://github.com/xavi-burgos99/gestur
Required Notice: Commercial licensing contact: xavi@dzin.es
```

CC BY-NC / BY-NC-SA may be appropriate for original documentation or creative
assets once ownership is established. They are not the preferred software
license: [Creative Commons advises against using its licenses for
software](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software).
A software-specific license should address the software grant and dependency
boundary directly.

## Scope and rights that cannot be withdrawn

A new license can govern only rights the maintainer controls. It cannot remove
MIT, BSD, Apache, LGPL, or other rights from dependency authors' code. It must not
relicense the operating system, separately installed tools, third-party model
weights, examples, or copied research code as if they were original Gestur code.
See [the audit](licensing-audit.md) and [notices](../THIRD_PARTY_NOTICES.md).

The Git history contains an earlier root Apache-2.0 license associated with an
OpenMMLab/MMPose import. Its intended scope is not established merely by the
copyright line in that file. Deleting it or adding a more restrictive license
now cannot cancel valid grants covering earlier copies. Review that history
before claiming exclusive control over commercial use of all versions or of
unchanged code already licensed to recipients.

The repository also identifies this as a university project. The maintainer
should confirm any agreements with the university, collaborators, employers,
funders, or asset providers, and the rights to the capitel scan and texture.
Being named as project developer does not by itself establish ownership of every
included asset. The audit identifies the actual evidence and unresolved cases;
it is not a warranty of rights clearance.

No copyright license can prohibit every conceivable benefit: it operates on the
rights the author owns, subject to statutory exceptions. It cannot monopolize
an idea or prevent independent implementation. Commercial permission should
specify the operator, intended deployment, duration, attribution, redistribution,
and any fee; it should not imply permission for excluded third-party material.

## Before adoption

1. Confirm whether attribution must be visible in public installations, or only
   preserved in copies and documentation. The draft assumes visible public credit.
2. Confirm the treatment of paid nonprofit exhibitions, sponsorship, and internal
   business use. The strict draft requires permission for these commercial uses.
3. Clear ownership and earlier licensing scope, and resolve the audit's release
   conditions for dependencies and model assets.
4. Have the chosen text reviewed for the applicable jurisdiction, then activate
   it deliberately with a version and effective source revision.

Contribution terms also need to preserve the ability to grant later commercial
licenses. A contributor's agreement must actually grant that authority; merely
accepting a contribution under a noncommercial license is not enough.

## Primary references

- [Open Source Definition](https://opensource.org/osd)
- [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
- [PolyForm text modification and naming rules](https://github.com/polyformproject/polyform-licenses/blob/1.0.0/README.md)
- [Creative Commons software guidance](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software)
- [Apache License 2.0, including the irrevocable copyright grant](https://www.apache.org/licenses/LICENSE-2.0)
- [GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.html)
